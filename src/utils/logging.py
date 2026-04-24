"""
Structured JSON logging for DriftSentinel.

All log records are emitted as JSON objects (structlog + stdlib logging).
In development mode (LOG_FORMAT=text), pretty-prints with colors instead.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a module-level structlog logger."""
    return structlog.get_logger(name)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """
    Configure structlog + stdlib logging.

    Call once at application startup (main.py / gunicorn entrypoint).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quieten noisy third-party loggers
    for noisy in ("kafka", "confluent_kafka", "pyspark", "py4j", "mlflow", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
