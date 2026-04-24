"""
DriftSentinel REST API

Endpoints:
  GET  /health                     Liveness + readiness
  GET  /metrics                    Prometheus scrape endpoint
  POST /models/{model_id}/reference   Register reference distribution
  GET  /models/{model_id}/status      Current drift status for a model
  GET  /alerts                     List recent drift alerts
  GET  /alerts/{alert_id}          Full alert + diagnosis report
  GET  /canary/{deployment_id}     Canary stage status
  POST /models/{model_id}/evaluate     Manually trigger drift evaluation on a batch
  GET  /reports/{diagnosis_id}     Full LLM diagnosis report
  GET  /models                     List monitored models
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

from src.drift.engine import DriftDetectionEngine, ReferenceStore
from src.ingestion.schema import (
    BatchFeatureStats,
    DriftAlert,
    DriftSeverity,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

DRIFT_ALERTS_TOTAL = Counter(
    "driftsentinel_alerts_total",
    "Total drift alerts fired",
    ["model_id", "severity"],
)
DRIFT_DETECTION_DURATION = Histogram(
    "driftsentinel_detection_duration_seconds",
    "Time to run drift detection on one batch",
    ["model_id"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
FEATURES_DRIFTED_GAUGE = Gauge(
    "driftsentinel_features_drifted",
    "Number of currently drifted features per model",
    ["model_id"],
)
CANARY_STAGE_GAUGE = Gauge(
    "driftsentinel_canary_stage_traffic_pct",
    "Current canary traffic percentage for challenger model",
    ["model_id", "challenger_version"],
)
MODELS_MONITORED = Gauge(
    "driftsentinel_models_monitored_total",
    "Number of model endpoints currently monitored",
)
API_REQUESTS_TOTAL = Counter(
    "driftsentinel_api_requests_total",
    "Total API requests",
    ["method", "path", "status_code"],
)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    reference_store: ReferenceStore = ReferenceStore()
    engine: DriftDetectionEngine = None
    alerts: list[DriftAlert] = []
    diagnosis_cache: dict[str, Any] = {}
    monitored_models: set[str] = set()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.engine = DriftDetectionEngine(reference_store=state.reference_store)
    MODELS_MONITORED.set(0)
    logger.info("api_startup_complete")
    yield
    logger.info("api_shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DriftSentinel API",
    description="Real-time ML observability — drift detection, diagnosis, and autonomous retraining",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request logging + metrics
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    API_REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
    ).inc()
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {
        "status": "healthy",
        "models_monitored": len(state.monitored_models),
        "alerts_in_memory": len(state.alerts),
    }


@app.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Reference registration
# ---------------------------------------------------------------------------

@app.post("/models/{model_id}/reference", tags=["models"])
async def register_reference(model_id: str, batch: BatchFeatureStats) -> dict:
    if batch.model_id != model_id:
        raise HTTPException(422, f"batch.model_id '{batch.model_id}' != path model_id '{model_id}'")

    state.engine.set_reference(batch)
    state.monitored_models.add(model_id)
    MODELS_MONITORED.set(len(state.monitored_models))

    logger.info("reference_registered", model_id=model_id, features=len(batch.features))
    return {"model_id": model_id, "features_registered": len(batch.features), "status": "ok"}


# ---------------------------------------------------------------------------
# Manual drift evaluation
# ---------------------------------------------------------------------------

@app.post("/models/{model_id}/evaluate", tags=["models"])
async def evaluate_batch(
    model_id: str,
    batch: BatchFeatureStats,
    background_tasks: BackgroundTasks,
) -> dict:
    if batch.model_id != model_id:
        raise HTTPException(422, "model_id mismatch")

    with DRIFT_DETECTION_DURATION.labels(model_id=model_id).time():
        alert = state.engine.evaluate(batch)

    if alert:
        state.alerts.append(alert)
        DRIFT_ALERTS_TOTAL.labels(
            model_id=model_id,
            severity=alert.severity.value,
        ).inc()
        FEATURES_DRIFTED_GAUGE.labels(model_id=model_id).set(
            len(alert.drifted_features)
        )
        return {
            "drift_detected": True,
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "drifted_features": alert.drifted_features,
            "tests_fired": alert.tests_fired,
        }
    else:
        FEATURES_DRIFTED_GAUGE.labels(model_id=model_id).set(0)
        return {"drift_detected": False}


# ---------------------------------------------------------------------------
# Model status
# ---------------------------------------------------------------------------

@app.get("/models", tags=["models"])
async def list_models() -> dict:
    return {
        "models": list(state.monitored_models),
        "total": len(state.monitored_models),
    }


@app.get("/models/{model_id}/status", tags=["models"])
async def model_status(model_id: str) -> dict:
    if model_id not in state.monitored_models:
        raise HTTPException(404, f"Model '{model_id}' not registered")

    recent_alerts = [
        a for a in state.alerts[-100:]
        if a.model_id == model_id
    ]
    return {
        "model_id": model_id,
        "monitored": True,
        "recent_alert_count": len(recent_alerts),
        "last_alert": recent_alerts[-1].model_dump() if recent_alerts else None,
        "severity_distribution": {
            s.value: sum(1 for a in recent_alerts if a.severity == s)
            for s in DriftSeverity
        },
    }


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@app.get("/alerts", tags=["alerts"])
async def list_alerts(
    limit: int = 50,
    model_id: str | None = None,
    severity: str | None = None,
) -> dict:
    alerts = state.alerts[-500:]   # Keep last 500 in memory; rest in persistent store

    if model_id:
        alerts = [a for a in alerts if a.model_id == model_id]
    if severity:
        alerts = [a for a in alerts if a.severity.value == severity]

    alerts = alerts[-limit:]
    return {
        "alerts": [a.model_dump() for a in alerts],
        "total": len(alerts),
    }


@app.get("/alerts/{alert_id}", tags=["alerts"])
async def get_alert(alert_id: str) -> dict:
    alert = next((a for a in state.alerts if a.alert_id == alert_id), None)
    if not alert:
        raise HTTPException(404, f"Alert '{alert_id}' not found")
    return alert.model_dump()


# ---------------------------------------------------------------------------
# Diagnosis reports
# ---------------------------------------------------------------------------

@app.get("/reports/{diagnosis_id}", tags=["diagnosis"])
async def get_report(diagnosis_id: str) -> dict:
    report = state.diagnosis_cache.get(diagnosis_id)
    if not report:
        raise HTTPException(404, f"Report '{diagnosis_id}' not found")
    return report


# ---------------------------------------------------------------------------
# Canary status
# ---------------------------------------------------------------------------

@app.get("/canary/{deployment_id}", tags=["canary"])
async def canary_status(deployment_id: str) -> dict:
    return {
        "deployment_id": deployment_id,
        "message": "Canary status tracked in Kafka canary-decisions topic and MLflow",
    }


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
