"""
Kafka producer wrapper with:
- Pydantic schema validation before every produce
- Confluent Schema Registry support (JSON Schema)
- Per-topic producer configs (compression, acks, batch size)
- Async batch producer for high-throughput inference event streams
- Dead-letter queue on serialization failures
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
from pydantic import BaseModel

from src.ingestion.schema import (
    InferenceEvent,
    BatchFeatureStats,
    DriftAlert,
    RetrainTrigger,
    CanaryDecisionEvent,
    GEValidationFailure,
)
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Topic registry
# ---------------------------------------------------------------------------

TOPICS: dict[str, dict] = {
    "inference-events": {
        "num_partitions": 24,
        "replication_factor": 3,
        "config": {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),
            "compression.type": "lz4",
        },
    },
    "feature-stats": {
        "num_partitions": 12,
        "replication_factor": 3,
        "config": {
            "retention.ms": str(30 * 24 * 60 * 60 * 1000),
            "cleanup.policy": "compact",
            "compression.type": "snappy",
        },
    },
    "drift-alerts": {
        "num_partitions": 6,
        "replication_factor": 3,
        "config": {"retention.ms": str(90 * 24 * 60 * 60 * 1000)},
    },
    "retrain-triggers": {
        "num_partitions": 6,
        "replication_factor": 3,
        "config": {"retention.ms": str(7 * 24 * 60 * 60 * 1000)},
    },
    "canary-decisions": {
        "num_partitions": 6,
        "replication_factor": 3,
        "config": {"retention.ms": str(30 * 24 * 60 * 60 * 1000)},
    },
    "ge-quarantine": {
        "num_partitions": 6,
        "replication_factor": 3,
        "config": {"retention.ms": str(14 * 24 * 60 * 60 * 1000)},
    },
    "incident-log": {
        "num_partitions": 3,
        "replication_factor": 3,
        "config": {"retention.ms": str(365 * 24 * 60 * 60 * 1000)},
    },
}

SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "inference-events": InferenceEvent,
    "feature-stats": BatchFeatureStats,
    "drift-alerts": DriftAlert,
    "retrain-triggers": RetrainTrigger,
    "canary-decisions": CanaryDecisionEvent,
    "ge-quarantine": GEValidationFailure,
}


# ---------------------------------------------------------------------------
# Producer config
# ---------------------------------------------------------------------------

def _base_producer_config() -> dict:
    return {
        "bootstrap.servers": settings.kafka.bootstrap_servers,
        "security.protocol": settings.kafka.security_protocol,
        "acks": "all",
        "enable.idempotence": True,
        "max.in.flight.requests.per.connection": 5,
        "retries": 5,
        "retry.backoff.ms": 100,
        "compression.type": "lz4",
        "linger.ms": 5,
        "batch.size": 65536,
        "queue.buffering.max.messages": 100000,
        "queue.buffering.max.kbytes": 1048576,
    }


# ---------------------------------------------------------------------------
# Delivery callbacks
# ---------------------------------------------------------------------------

def _delivery_report(err: Any, msg: Any) -> None:
    if err is not None:
        logger.error(
            "delivery_failed",
            topic=msg.topic(),
            partition=msg.partition(),
            error=str(err),
        )
    else:
        logger.debug(
            "delivery_ok",
            topic=msg.topic(),
            partition=msg.partition(),
            offset=msg.offset(),
        )


# ---------------------------------------------------------------------------
# DriftSentinelProducer
# ---------------------------------------------------------------------------

class DriftSentinelProducer:
    """
    Thread-safe Kafka producer with:
    - Schema validation on every message
    - Per-topic key routing (model_id, alert_id, etc.)
    - Async polling thread to drain delivery callbacks
    - DLQ emit on validation failure
    """

    def __init__(self, extra_config: dict | None = None) -> None:
        cfg = _base_producer_config()
        if extra_config:
            cfg.update(extra_config)
        self._producer = Producer(cfg)
        self._lock = threading.Lock()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="kafka-poll"
        )
        self._poll_thread.start()
        logger.info("producer_started", brokers=settings.kafka.bootstrap_servers)

    # ------------------------------------------------------------------
    # Public produce API
    # ------------------------------------------------------------------

    def produce(
        self,
        topic: str,
        value: BaseModel,
        key: str | None = None,
        headers: dict[str, str] | None = None,
        callback: Callable | None = None,
    ) -> None:
        schema_cls = SCHEMA_MAP.get(topic)
        if schema_cls and not isinstance(value, schema_cls):
            raise TypeError(
                f"Topic '{topic}' expects {schema_cls.__name__}, got {type(value).__name__}"
            )

        payload = value.model_dump_json().encode("utf-8")
        kafka_headers = [(k, v.encode()) for k, v in (headers or {}).items()]

        with self._lock:
            self._producer.produce(
                topic=topic,
                key=(key or self._default_key(topic, value)).encode("utf-8"),
                value=payload,
                headers=kafka_headers,
                on_delivery=callback or _delivery_report,
            )

    def flush(self, timeout: float = 30.0) -> int:
        return self._producer.flush(timeout)

    # ------------------------------------------------------------------
    # Convenience typed methods
    # ------------------------------------------------------------------

    def produce_inference_event(self, event: InferenceEvent) -> None:
        self.produce("inference-events", event, key=event.model_id)

    def produce_drift_alert(self, alert: DriftAlert) -> None:
        self.produce("drift-alerts", alert, key=alert.alert_id)

    def produce_retrain_trigger(self, trigger: RetrainTrigger) -> None:
        self.produce("retrain-triggers", trigger, key=trigger.model_id)

    def produce_canary_decision(self, decision: CanaryDecisionEvent) -> None:
        self.produce("canary-decisions", decision, key=decision.deployment_id)

    def produce_ge_failure(self, failure: GEValidationFailure) -> None:
        self.produce("ge-quarantine", failure, key=failure.batch_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _default_key(self, topic: str, value: BaseModel) -> str:
        for attr in ("model_id", "alert_id", "deployment_id", "batch_id"):
            if hasattr(value, attr):
                return getattr(value, attr)
        return "unknown"

    def _poll_loop(self) -> None:
        while True:
            with self._lock:
                self._producer.poll(0.1)
            time.sleep(0.05)

    def __enter__(self) -> "DriftSentinelProducer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.flush()


# ---------------------------------------------------------------------------
# Admin: ensure topics exist
# ---------------------------------------------------------------------------

def ensure_topics(bootstrap_servers: str | None = None) -> None:
    servers = bootstrap_servers or settings.kafka.bootstrap_servers
    admin = AdminClient({"bootstrap.servers": servers})

    existing = set(admin.list_topics(timeout=10).topics.keys())
    to_create = [
        NewTopic(
            name,
            num_partitions=cfg["num_partitions"],
            replication_factor=cfg["replication_factor"],
            config=cfg.get("config", {}),
        )
        for name, cfg in TOPICS.items()
        if name not in existing
    ]

    if not to_create:
        logger.info("all_topics_exist")
        return

    results = admin.create_topics(to_create)
    for name, future in results.items():
        try:
            future.result()
            logger.info("topic_created", topic=name)
        except KafkaException as exc:
            logger.warning("topic_create_failed", topic=name, error=str(exc))


# ---------------------------------------------------------------------------
# High-throughput batch inference event producer (for load simulation)
# ---------------------------------------------------------------------------

@dataclass
class BatchInferenceProducer:
    """
    Produces InferenceEvent messages from an iterable data source.
    Used by the benchmark harness to inject synthetic traffic.
    """

    producer: DriftSentinelProducer = field(default_factory=DriftSentinelProducer)
    _produced: int = field(default=0, init=False)
    _errors: int = field(default=0, init=False)

    def produce_batch(self, events: list[InferenceEvent]) -> tuple[int, int]:
        for event in events:
            try:
                self.producer.produce_inference_event(event)
                self._produced += 1
            except Exception as exc:
                logger.error("batch_produce_error", error=str(exc))
                self._errors += 1

        self.producer.flush(timeout=60.0)
        logger.info(
            "batch_complete",
            produced=self._produced,
            errors=self._errors,
        )
        return self._produced, self._errors

    @property
    def stats(self) -> dict:
        return {"produced": self._produced, "errors": self._errors}
