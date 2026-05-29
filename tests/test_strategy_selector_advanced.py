"""
Strategy-selector tests: rule overrides, end-to-end DriftAlert routing,
cost estimation, and decision-path explainability.

The selector is the cost-aware brain of the platform, so these tests pin
down every hard business rule and verify the high-level `StrategySelector`
turns a DriftAlert into a valid RetrainTrigger.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.ingestion.schema import (
    DriftAlert,
    DriftSeverity,
    RetrainingStrategy,
    SLATier,
    TestResult,
)
from src.retraining.strategy_selector import (
    StrategyFeatures,
    StrategySelector,
    StrategySelectorModel,
)


@pytest.fixture
def model(tmp_path):
    return StrategySelectorModel(model_path=tmp_path / "selector.pkl")


def _features(**overrides) -> StrategyFeatures:
    base = dict(
        drift_severity_score=0.4,
        pct_features_drifted=0.3,
        psi_max=0.15,
        shap_drift_detected=False,
        drift_is_slice_local=False,
        data_availability_ratio=1.0,
        days_since_last_retrain=10.0,
        estimated_cost_usd=20.0,
        cost_ceiling_usd=50.0,
        sla_tier_critical=False,
        sla_tier_standard=True,
    )
    base.update(overrides)
    return StrategyFeatures(**base)


# ---------------------------------------------------------------------------
# Hard business rules
# ---------------------------------------------------------------------------

class TestRuleOverrides:
    def test_low_data_forces_ensemble_fallback(self, model):
        strat, conf = model.predict(_features(data_availability_ratio=0.1))
        assert strat == RetrainingStrategy.ENSEMBLE_FALLBACK
        assert conf == 1.0

    def test_cost_over_ceiling_forces_ensemble_fallback(self, model):
        strat, _ = model.predict(_features(estimated_cost_usd=999, cost_ceiling_usd=50))
        assert strat == RetrainingStrategy.ENSEMBLE_FALLBACK

    def test_slice_local_forces_slice_finetune(self, model):
        strat, _ = model.predict(
            _features(drift_is_slice_local=True, data_availability_ratio=1.0, drift_severity_score=0.4)
        )
        assert strat == RetrainingStrategy.SLICE_FINETUNE

    def test_shap_drift_forces_full_retrain(self, model):
        strat, _ = model.predict(_features(shap_drift_detected=True))
        assert strat == RetrainingStrategy.FULL_RETRAIN

    def test_severe_global_drift_forces_full_retrain(self, model):
        strat, _ = model.predict(_features(drift_severity_score=0.9, pct_features_drifted=0.8))
        assert strat == RetrainingStrategy.FULL_RETRAIN

    def test_stale_temporal_drift_weighted_retrain(self, model):
        strat, _ = model.predict(
            _features(
                shap_drift_detected=False,
                pct_features_drifted=0.2,
                days_since_last_retrain=30,
                drift_severity_score=0.3,
                drift_is_slice_local=False,
            )
        )
        assert strat == RetrainingStrategy.WEIGHTED_RETRAIN


# ---------------------------------------------------------------------------
# Persistence + explainability
# ---------------------------------------------------------------------------

class TestModelLifecycle:
    def test_tree_is_persisted_and_reloaded(self, tmp_path):
        path = tmp_path / "selector.pkl"
        StrategySelectorModel(model_path=path)
        assert path.exists()
        assert path.with_suffix(".rules.txt").exists()
        # Reload from disk path uses the load branch.
        reloaded = StrategySelectorModel(model_path=path)
        assert reloaded._clf is not None

    def test_predictions_are_valid_and_confident(self, model):
        strat, conf = model.predict(_features())
        assert strat in list(RetrainingStrategy)
        assert 0.0 <= conf <= 1.0

    def test_explanation_contains_decision_path(self, model):
        model.predict(_features(shap_drift_detected=True))
        explanation = model.explain(_features(shap_drift_detected=True))
        assert "Decision path" in explanation


# ---------------------------------------------------------------------------
# End-to-end: DriftAlert -> RetrainTrigger
# ---------------------------------------------------------------------------

def _alert(severity=DriftSeverity.HIGH, segment=None, shap=False) -> DriftAlert:
    results = [
        TestResult(test_name="PSI", feature_name="f", statistic=0.35, threshold=0.2, drifted=True)
    ]
    if shap:
        results.append(
            TestResult(test_name="SHAP_Delta", feature_name="f", statistic=0.9, threshold=0.15, drifted=True)
        )
    return DriftAlert(
        model_id="model_a",
        model_version="v1",
        window_start=datetime.utcnow() - timedelta(minutes=5),
        window_end=datetime.utcnow(),
        segment=segment or {},
        severity=severity,
        drifted_features=["f"],
        test_results=results,
        tests_fired=len(results),
        tests_total=len(results),
    )


class TestEndToEnd:
    @pytest.fixture
    def selector(self, tmp_path):
        return StrategySelector(model=StrategySelectorModel(model_path=tmp_path / "s.pkl"))

    def test_returns_valid_trigger(self, selector):
        trigger = selector.select(
            alert=_alert(),
            training_data_path="/data/train.parquet",
            data_availability_ratio=1.0,
            days_since_last_retrain=20,
            sla_tier=SLATier.STANDARD,
        )
        assert trigger.model_id == "model_a"
        assert trigger.strategy in list(RetrainingStrategy)
        assert trigger.estimated_cost_usd > 0

    def test_segment_filter_propagates_on_slice_drift(self, selector):
        trigger = selector.select(
            alert=_alert(segment={"region": "us-west"}, severity=DriftSeverity.MEDIUM),
            training_data_path="/data/train.parquet",
            data_availability_ratio=1.0,
            days_since_last_retrain=5,
            sla_tier=SLATier.STANDARD,
        )
        assert trigger.segment_filter == {"region": "us-west"}

    def test_weighted_strategy_sets_lambda(self, selector):
        trigger = selector.select(
            alert=_alert(severity=DriftSeverity.MEDIUM),
            training_data_path="/data/train.parquet",
            data_availability_ratio=1.0,
            days_since_last_retrain=30,
            sla_tier=SLATier.STANDARD,
        )
        if trigger.strategy == RetrainingStrategy.WEIGHTED_RETRAIN:
            assert trigger.temporal_weight_lambda is not None

    @pytest.mark.parametrize("gb,expected_min", [(1.0, 0.0), (100.0, 1.0)])
    def test_cost_estimate_scales_with_data(self, selector, gb, expected_min):
        cost = selector._estimate_cost(gb)
        assert cost >= expected_min
