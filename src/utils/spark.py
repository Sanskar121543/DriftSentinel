"""
PySpark session factory for DriftSentinel.

Creates a SparkSession configured for Kafka structured streaming with
sensible defaults for DriftSentinel's workload profile.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from src.utils.config import settings


def get_or_create_spark(app_name: str | None = None) -> SparkSession:
    """
    Return (or create) a SparkSession with Kafka + DriftSentinel config.

    Call once per process.  Spark handles internal singleton management.
    """
    cfg = settings.spark
    name = app_name or cfg.app_name

    builder = (
        SparkSession.builder
        .appName(name)
        .master(cfg.master)
        .config("spark.executor.memory", cfg.executor_memory)
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.cores", str(cfg.executor_cores))
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        # Kafka structured streaming
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
            "org.apache.kafka:kafka-clients:3.7.0",
        )
        # Streaming optimizations
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.streaming.checkpointLocation", cfg.checkpoint_path)
        # Avoid OOM on large micro-batches
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Reduce verbose Spark logging
        .config("spark.ui.showConsoleProgress", "false")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
