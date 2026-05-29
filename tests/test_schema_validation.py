"""
Pydantic schema validation and serialization tests.

Every event crossing a Kafka topic is validated against these models, so the
suite pins down: required-field enforcement, the features-not-empty validator,
enum coercion, default factories (UUIDs, timestamps), and JSON round-trips
for the richer nested events.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from src.ingestion.schema import (
    CanaryDecision,
    CanaryDecisionEvent,
    CanaryStageMetrics,
    DriftAlert,
    DriftSeverity,
    InferenceEvent,
    RetrainingStrategy,
    RetrainTrigger,
    SLATier,
    TestResult,
)


# ---------------------------------------------------------------------------
# InferenceEvent
# ---------------------------------------------------------------------------

class TestInferenceEvent:
    def test_minimal_valid(self):
        ev = InferenceEvent(
            model_id="m",
            model_version="v1",
            features={"x": 1},
            prediction=0.5,
            latency_ms=10.0,
        )
        assert ev.event_id  # default uuid
        assert isinstance(ev.timestamp, datetime)

    def test_empty_features_rejected(self):
        with pytest.raises(ValidationError):
            InferenceEvent(
                model_id="m",
                model_version="v1",
                features={},
                prediction=0.5,
                latency_ms=10.0,
            )

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            InferenceEvent(model_id="m", features={"x": 1}, prediction=1, latency_ms=1.0)

    def test_unique_event_ids(self):
        a = InferenceEvent(model_id="m", model_version="v", features={"x": 1}, prediction=1, latency_ms=1)
        b = InferenceEvent(model_id="m", model_version="v", features={"x": 1}, prediction=1, latency_ms=1)
        assert a.event_id != b.event_id

    @pytest.mark.parametrize("prediction", [0.5, 1, "fraud", [0.1, 0.9]])
    def test_prediction_union_types(self, prediction):
        ev = InferenceEvent(
            model_id="m", model_version="v", features={"x": 1},
            prediction=prediction, latency_ms=1.0,
        )
        assert ev.prediction == prediction


# ---------------------------------------------------------------------------
# Enum coercion
# ---------------------------------------------------------------------------

class TestEnums:
    def test_severity_from_string(self):
        alert = _make_alert(severity="critical")
        assert alert.severity == DriftSeverity.CRITICAL

    def test_invalid_enum_rejected(self):
        with pytest.raises(ValidationError):
            _make_alert(severity="apocalyptic")

    def test_strategy_enum_values(self):
        assert {s.value for s in RetrainingStrategy} == {
            "full_retrain", "weighted_retrain", "slice_finetune", "ensemble_fallback",
        }


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------

class TestRoundTrips:
    def test_drift_alert_roundtrip(self):
        alert = _make_alert()
        reloaded = DriftAlert.model_validate_json(alert.model_dump_json())
        assert reloaded.alert_id == alert.alert_id
        assert reloaded.severity == alert.severity

    def test_retrain_trigger_roundtrip(self):
        trigger = RetrainTrigger(
            alert_id="a1",
            model_id="m",
            strategy=RetrainingStrategy.WEIGHTED_RETRAIN,
            estimated_cost_usd=12.5,
            sla_tier=SLATier.CRITICAL,
            training_data_path="/data",
            temporal_weight_lambda=0.05,
        )
        reloaded = RetrainTrigger.model_validate_json(trigger.model_dump_json())
        assert reloaded.strategy == RetrainingStrategy.WEIGHTED_RETRAIN
        assert reloaded.temporal_weight_lambda == 0.05

    def test_canary_decision_event_with_stages_roundtrip(self):
        stage = CanaryStageMetrics(
            stage_traffic_pct=0.05,
            sample_size=500,
            prediction_quality=0.91,
            p50_latency_ms=20.0,
            p99_latency_ms=120.0,
            error_rate=0.002,
            business_metric=0.31,
            business_metric_name="conversion",
            sprt_llr=2.4,
            sprt_decision=CanaryDecision.PROMOTE,
        )
        event = CanaryDecisionEvent(
            deployment_id="d1",
            model_id="m",
            challenger_version="v2",
            champion_version="v1",
            final_decision=CanaryDecision.PROMOTE,
            stage_history=[stage],
        )
        reloaded = CanaryDecisionEvent.model_validate_json(event.model_dump_json())
        assert len(reloaded.stage_history) == 1
        assert reloaded.stage_history[0].sprt_decision == CanaryDecision.PROMOTE


def _make_alert(severity="high") -> DriftAlert:
    return DriftAlert(
        model_id="m",
        model_version="v1",
        window_start=datetime.utcnow() - timedelta(minutes=5),
        window_end=datetime.utcnow(),
        severity=severity,
        drifted_features=["f"],
        test_results=[
            TestResult(test_name="KS", feature_name="f", statistic=0.3, p_value=0.001, threshold=0.05, drifted=True)
        ],
        tests_fired=1,
        tests_total=1,
    )
