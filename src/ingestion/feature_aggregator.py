"""
Spark Structured Streaming job: Feature Distribution Aggregator

Reads from inference-events Kafka topic, computes per-feature distribution
statistics over 5-minute tumbling windows (segmented by model_id + slice keys),
and writes results to feature-stats Kafka topic.

Design decisions:
  - Stateless aggregation: rolling stats stored in Kafka compacted topic,
    not Spark driver memory → crash-safe for 24/7 operation
  - Watermarking: 10-min late-data tolerance before window is finalized
  - UDFs for percentile computation (approxQuantile via DataFrameStatFunctions)
  - SHAP values optionally joined from a side-stream if provided
  - Output mode: append (only emit finalized windows)
"""

from __future__ import annotations

import json

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    ArrayType, MapType, LongType, TimestampType,
)

from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.spark import get_or_create_spark

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Inference event schema for Spark
# ---------------------------------------------------------------------------

INFERENCE_EVENT_SCHEMA = StructType([
    StructField("event_id", StringType()),
    StructField("model_id", StringType()),
    StructField("model_version", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("features", MapType(StringType(), StringType())),   # JSON-parsed downstream
    StructField("prediction", StringType()),
    StructField("prediction_proba", ArrayType(DoubleType())),
    StructField("latency_ms", DoubleType()),
    StructField("segment", MapType(StringType(), StringType())),
    StructField("label", StringType()),
])


# ---------------------------------------------------------------------------
# Feature aggregator job
# ---------------------------------------------------------------------------

class FeatureAggregator:
    """
    Reads inference-events, aggregates feature stats per 5-minute window,
    segments by model_id + configured slice keys, writes to feature-stats.
    """

    def __init__(self, spark: SparkSession | None = None) -> None:
        self.spark = spark or get_or_create_spark("DriftSentinel-FeatureAgg")
        self.window_duration = settings.drift.window_minutes
        self.watermark_duration = f"{settings.drift.watermark_minutes} minutes"
        self.kafka_cfg = settings.kafka

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    def _read_inference_stream(self) -> DataFrame:
        raw = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_cfg.bootstrap_servers)
            .option("subscribe", "inference-events")
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .option("kafka.group.id", "feature-aggregator")
            .option("maxOffsetsPerTrigger", 500_000)
            .load()
        )

        parsed = (
            raw.select(
                F.col("key").cast(StringType()).alias("_kafka_key"),
                F.from_json(
                    F.col("value").cast(StringType()),
                    INFERENCE_EVENT_SCHEMA,
                ).alias("e"),
                F.col("timestamp").alias("_kafka_ts"),
            )
            .select("e.*", "_kafka_key")
            .withWatermark("timestamp", self.watermark_duration)
        )

        return parsed

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_window(self, stream: DataFrame) -> DataFrame:
        """
        Tumbling window aggregation.
        For each (model_id, window, segment) group:
          - numeric features: mean, std, min, max, approx percentiles
          - string features handled via value_counts approximation
          - null rates tracked per feature
        """
        window_col = F.window("timestamp", f"{self.window_duration} minutes")

        # Explode the features map so each (event, feature) becomes one row
        exploded = stream.select(
            "model_id",
            "model_version",
            "timestamp",
            "segment",
            F.explode("features").alias("feature_name", "feature_value_str"),
        ).withColumn(
            "feature_value_num",
            F.col("feature_value_str").cast(DoubleType()),
        ).withColumn(
            "is_null",
            F.col("feature_value_str").isNull() | F.col("feature_value_num").isNull(),
        )

        # Segment key: flatten the map to a string for grouping
        exploded = exploded.withColumn(
            "segment_key",
            F.to_json(F.col("segment")),
        )

        agg = (
            exploded
            .groupBy(
                "model_id",
                "feature_name",
                "segment_key",
                window_col,
            )
            .agg(
                F.count("*").alias("total_count"),
                F.sum(F.col("is_null").cast(LongType())).alias("null_count"),
                F.mean("feature_value_num").alias("mean"),
                F.stddev("feature_value_num").alias("std"),
                F.min("feature_value_num").alias("min_val"),
                F.max("feature_value_num").alias("max_val"),
                # Approximate percentiles — faster than exact on large streams
                F.percentile_approx("feature_value_num", [0.25, 0.5, 0.75, 0.95, 0.99]).alias("pcts"),
                # Categorical: count distinct string values (bounded cardinality)
                F.collect_list(
                    F.when(F.col("feature_value_num").isNull(), F.col("feature_value_str"))
                ).alias("cat_values"),
                F.first("model_version").alias("model_version"),
            )
        )

        # Reshape pcts array into named columns
        result = (
            agg
            .withColumn("p25", F.col("pcts")[0])
            .withColumn("p50", F.col("pcts")[1])
            .withColumn("p75", F.col("pcts")[2])
            .withColumn("p95", F.col("pcts")[3])
            .withColumn("p99", F.col("pcts")[4])
            .drop("pcts")
            .withColumn("window_start", F.col("window.start"))
            .withColumn("window_end", F.col("window.end"))
            .drop("window")
        )

        return result

    # ------------------------------------------------------------------
    # Sink
    # ------------------------------------------------------------------

    def _serialize_to_kafka(self, row: dict) -> dict:
        """Convert aggregated row dict to BatchFeatureStats-compatible JSON."""
        segment = json.loads(row.get("segment_key") or "{}")
        cat_values = row.get("cat_values") or []

        value_counts: dict[str, int] | None = None
        if cat_values:
            counts: dict[str, int] = {}
            for v in cat_values:
                if v is not None:
                    counts[str(v)] = counts.get(str(v), 0) + 1
            value_counts = counts if counts else None

        feature_type = "categorical" if value_counts else "continuous"

        feature_stat = {
            "feature_name": row["feature_name"],
            "feature_type": feature_type,
            "model_id": row["model_id"],
            "window_start": row["window_start"].isoformat() if row.get("window_start") else None,
            "window_end": row["window_end"].isoformat() if row.get("window_end") else None,
            "segment": segment,
            "total_count": row.get("total_count", 0),
            "null_count": row.get("null_count", 0),
            "mean": row.get("mean"),
            "std": row.get("std"),
            "min": row.get("min_val"),
            "max": row.get("max_val"),
            "p25": row.get("p25"),
            "p50": row.get("p50"),
            "p75": row.get("p75"),
            "p95": row.get("p95"),
            "p99": row.get("p99"),
            "value_counts": value_counts,
        }
        return feature_stat

    def _write_to_kafka(self, df: DataFrame) -> None:
        """Write aggregated stats back to feature-stats topic."""
        kafka_df = df.select(
            F.col("model_id").alias("key"),
            F.to_json(F.struct("*")).alias("value"),
        )

        query = (
            kafka_df.writeStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_cfg.bootstrap_servers)
            .option("topic", "feature-stats")
            .option("checkpointLocation", settings.spark.checkpoint_path + "/feature-agg")
            .outputMode("append")
            .trigger(processingTime=f"{self.window_duration} minutes")
            .start()
        )

        logger.info("feature_agg_stream_started", trigger=f"{self.window_duration}m")
        return query

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        stream = self._read_inference_stream()
        agg = self._aggregate_window(stream)
        query = self._write_to_kafka(agg)

        try:
            query.awaitTermination()
        except KeyboardInterrupt:
            logger.info("feature_agg_stopping")
            query.stop()


def main() -> None:
    agg = FeatureAggregator()
    agg.run()


if __name__ == "__main__":
    main()
