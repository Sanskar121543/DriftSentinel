# DriftSentinel Dockerfile
# Multi-stage build: base → api | worker
#
# Usage:
#   docker build --target api -t driftsentinel-api .
#   docker build --target worker -t driftsentinel-worker .

# ─── Base ──────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for confluent-kafka, scipy, matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ librdkafka-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ─── API ───────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ─── Worker (diagnosis agent + strategy selector) ──────────
FROM base AS worker

CMD ["python", "-m", "src.workers.diagnosis_worker"]

# ─── Spark feature aggregator ──────────────────────────────
FROM base AS spark

RUN pip install --no-cache-dir pyspark==3.5.1

CMD ["python", "-m", "src.ingestion.feature_aggregator"]
