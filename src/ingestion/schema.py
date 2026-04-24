"""
Pydantic schemas for all Kafka events flowing through DriftSentinel.

Every message on every topic is validated against one of these schemas before
being produced or after being consumed.  Using Pydantic v2 for speed and
schema-export for Schema Registry registration.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FeatureType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"


class DriftSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetrainingStrategy(str, Enum):
    FULL_RETRAIN = "full_retrain"
    WEIGHTED_RETRAIN = "weighted_retrain"
    SLICE_FINETUNE = "slice_finetune"
    ENSEMBLE_FALLBACK = "ensemble_fallback"


class CanaryDecision(str, Enum):
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


class SLATier(str, Enum):
    CRITICAL = "critical"
    STANDARD = "standard"
    EXPERIMENTAL = "experimental"


# ---------------------------------------------------------------------------
# Topic: inference-events
# ---------------------------------------------------------------------------


class InferenceEvent(BaseModel):
    """One prediction request/response pair emitted by a model serving layer."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    features: dict[str, Any]
    prediction: float | int | str | list[float]
    prediction_proba: list[float] | None = None
    latency_ms: float
    segment: dict[str, str] = Field(
        default_factory=dict,
        description="Slice keys: {'platform': 'mobile', 'region': 'us-west'}",
    )
    label: float | int | str | None = Field(
        None,
        description="Ground-truth label if available (delayed feedback loop)",
    )

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("features dict must not be empty")
        return v


# ---------------------------------------------------------------------------
# Topic: feature-stats
# ---------------------------------------------------------------------------


class FeatureDistributionStats(BaseModel):
    """Aggregated statistics for one feature over one micro-batch window."""

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    feature_name: str
    feature_type: FeatureType
    window_start: datetime
    window_end: datetime
    segment: dict[str, str] = Field(default_factory=dict)

    # Continuous
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None
    p99: float | None = None
    histogram_edges: list[float] | None = None
    histogram_counts: list[int] | None = None

    # Categorical
    value_counts: dict[str, int] | None = None
    cardinality: int | None = None

    # Common
    null_count: int = 0
    total_count: int
    shap_mean_abs: float | None = None


class BatchFeatureStats(BaseModel):
    """All feature stats for one model over one window — single Kafka message."""

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    window_start: datetime
    window_end: datetime
    segment: dict[str, str] = Field(default_factory=dict)
    features: list[FeatureDistributionStats]
    passed_ge_validation: bool = True
    quarantined_count: int = 0


# ---------------------------------------------------------------------------
# Topic: drift-alerts
# ---------------------------------------------------------------------------


class TestResult(BaseModel):
    test_name: str
    feature_name: str
    statistic: float
    p_value: float | None = None
    threshold: float
    drifted: bool
    details: dict[str, Any] = Field(default_factory=dict)


class DriftAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_version: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    window_start: datetime
    window_end: datetime
    segment: dict[str, str] = Field(default_factory=dict)

    severity: DriftSeverity
    drifted_features: list[str]
    test_results: list[TestResult]
    tests_fired: int
    tests_total: int

    # Populated async by LLM agent
    diagnosis_id: str | None = None
    root_cause_hypothesis: str | None = None
    recommended_strategy: RetrainingStrategy | None = None
    estimated_impact_usd: float | None = None


# ---------------------------------------------------------------------------
# Topic: retrain-triggers
# ---------------------------------------------------------------------------


class RetrainTrigger(BaseModel):
    trigger_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str
    model_id: str
    strategy: RetrainingStrategy
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_cost_usd: float
    sla_tier: SLATier
    training_data_path: str
    segment_filter: dict[str, str] | None = None
    temporal_weight_lambda: float | None = Field(
        None,
        description="Exponential decay lambda for weighted retraining",
    )
    airflow_dag_run_id: str | None = None


# ---------------------------------------------------------------------------
# Topic: canary-decisions
# ---------------------------------------------------------------------------


class CanaryStageMetrics(BaseModel):
    stage_traffic_pct: float
    sample_size: int
    prediction_quality: float
    p50_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    business_metric: float
    business_metric_name: str
    sprt_llr: float = Field(description="Log-likelihood ratio from SPRT")
    sprt_decision: CanaryDecision


class CanaryDecisionEvent(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    deployment_id: str
    model_id: str
    challenger_version: str
    champion_version: str
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    final_decision: CanaryDecision
    stage_history: list[CanaryStageMetrics]
    rollback_reason: str | None = None
    jira_ticket_id: str | None = None


# ---------------------------------------------------------------------------
# Topic: ge-quarantine
# ---------------------------------------------------------------------------


class GEValidationFailure(BaseModel):
    failure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    batch_id: str
    model_id: str
    failed_at: datetime = Field(default_factory=datetime.utcnow)
    expectation_suite: str
    failed_expectations: list[dict[str, Any]]
    quarantined_record_count: int
    total_record_count: int
    raw_results_path: str | None = None


# ---------------------------------------------------------------------------
# Incident report (written to Pinecone + MLflow)
# ---------------------------------------------------------------------------


class DiagnosisReport(BaseModel):
    diagnosis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str
    model_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    similar_incidents: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]] = Field(
        description="[{hypothesis: str, confidence: float, evidence: list[str]}]"
    )
    top_hypothesis: str
    top_hypothesis_confidence: float
    affected_segments: list[dict[str, str]]
    recommended_strategy: RetrainingStrategy
    strategy_rationale: str
    estimated_impact_usd: float
    full_report_markdown: str
    shap_plot_paths: list[str] = Field(default_factory=list)
    embedding: list[float] | None = Field(
        None, description="Stored in Pinecone for future RAG retrieval"
    )
