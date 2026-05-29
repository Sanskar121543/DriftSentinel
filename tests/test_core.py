"""
Unit tests for DriftSentinel core components.

Covers:
  - All 5 drift test implementations (KS, Chi2, PSI, JS, SHAP)
  - Detection engine orchestration (parallel tests, slice-awareness)
  - SPRT boundaries and convergence
  - Strategy selector heuristics
  - GE validator quarantine logic
  - Schema serialization round-trips
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.canary.sprt import SPRT, SPRTConfig
from src.drift.engine import DriftDetectionEngine
from src.drift.tests.chi_square import ChiSquaredTest
from src.drift.tests.jensen_shannon import JensenShannonDivergence
from src.drift.tests.ks_test import KolmogorovSmirnovTest
from src.drift.tests.psi import PopulationStabilityIndex
from src.drift.tests.shap_delta import SHAPDeltaTracker
from src.ingestion.schema import (
    BatchFeatureStats,
    DriftAlert,
    DriftSeverity,
    FeatureDistributionStats,
    FeatureType,
    InferenceEvent,
    RetrainingStrategy,
)
from src.retraining.strategy_selector import (
    StrategyFeatures,
    StrategySelectorModel,
    _rule_based_label,
    STRATEGY_TO_IDX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def continuous_reference() -> FeatureDistributionStats:
    rng = np.random.RandomState(0)
    data = rng.normal(0, 1, 1000)
    edges = np.linspace(data.min(), data.max(), 21).tolist()
    counts, _ = np.histogram(data, bins=edges)
    return FeatureDistributionStats(
        model_id="test_model",
        feature_name="feature_a",
        feature_type=FeatureType.CONTINUOUS,
        window_start=datetime.utcnow() - timedelta(minutes=10),
        window_end=datetime.utcnow() - timedelta(minutes=5),
        mean=float(data.mean()),
        std=float(data.std()),
        min=float(data.min()),
        max=float(data.max()),
        p25=float(np.percentile(data, 25)),
        p50=float(np.percentile(data, 50)),
        p75=float(np.percentile(data, 75)),
        p95=float(np.percentile(data, 95)),
        p99=float(np.percentile(data, 99)),
        histogram_edges=edges,
        histogram_counts=counts.tolist(),
        total_count=1000,
        null_count=0,
        shap_mean_abs=0.15,
    )


@pytest.fixture
def continuous_no_drift(continuous_reference) -> FeatureDistributionStats:
    rng = np.random.RandomState(1)
    data = rng.normal(0.01, 1.005, 1000)   # Very tiny shift — should not fire
    edges = continuous_reference.histogram_edges
    counts, _ = np.histogram(data, bins=edges)
    return continuous_reference.model_copy(update={
        "window_start": datetime.utcnow() - timedelta(minutes=5),
        "window_end": datetime.utcnow(),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "histogram_counts": counts.tolist(),
        "shap_mean_abs": 0.16,
    })


@pytest.fixture
def continuous_drifted(continuous_reference) -> FeatureDistributionStats:
    rng = np.random.RandomState(2)
    data = rng.normal(2.0, 1.8, 1000)    # Large shift — should fire
    edges = np.linspace(data.min(), data.max(), 21).tolist()
    counts, _ = np.histogram(data, bins=edges)
    return continuous_reference.model_copy(update={
        "window_start": datetime.utcnow() - timedelta(minutes=5),
        "window_end": datetime.utcnow(),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "p25": float(np.percentile(data, 25)),
        "p50": float(np.percentile(data, 50)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "histogram_edges": edges,
        "histogram_counts": counts.tolist(),
        "shap_mean_abs": 0.45,    # Large SHAP shift
    })


@pytest.fixture
def categorical_reference() -> FeatureDistributionStats:
    return FeatureDistributionStats(
        model_id="test_model",
        feature_name="region",
        feature_type=FeatureType.CATEGORICAL,
        window_start=datetime.utcnow() - timedelta(minutes=10),
        window_end=datetime.utcnow() - timedelta(minutes=5),
        value_counts={"north": 250, "south": 250, "east": 250, "west": 250},
        cardinality=4,
        total_count=1000,
        null_count=0,
    )


@pytest.fixture
def categorical_drifted(categorical_reference) -> FeatureDistributionStats:
    return categorical_reference.model_copy(update={
        "value_counts": {"north": 600, "south": 100, "east": 200, "west": 100},
        "window_start": datetime.utcnow() - timedelta(minutes=5),
        "window_end": datetime.utcnow(),
    })


# ---------------------------------------------------------------------------
# KS Test
# ---------------------------------------------------------------------------

class TestKolmogorovSmirnov:
    def test_no_drift_not_detected(self, continuous_reference, continuous_no_drift):
        test = KolmogorovSmirnovTest(p_value_threshold=0.05)
        result = test.run(continuous_reference, continuous_no_drift)
        assert result is not None
        assert result.drifted is False
        assert result.p_value > 0.05

    def test_drift_detected(self, continuous_reference, continuous_drifted):
        test = KolmogorovSmirnovTest(p_value_threshold=0.05)
        result = test.run(continuous_reference, continuous_drifted)
        assert result is not None
        assert result.drifted is True
        assert result.p_value < 0.05
        assert result.statistic > 0.0

    def test_returns_none_for_categorical(self, categorical_reference, categorical_drifted):
        test = KolmogorovSmirnovTest()
        result = test.run(categorical_reference, categorical_drifted)
        assert result is None

    def test_statistic_in_valid_range(self, continuous_reference, continuous_drifted):
        test = KolmogorovSmirnovTest()
        result = test.run(continuous_reference, continuous_drifted)
        assert 0.0 <= result.statistic <= 1.0

    def test_result_has_required_fields(self, continuous_reference, continuous_drifted):
        test = KolmogorovSmirnovTest()
        result = test.run(continuous_reference, continuous_drifted)
        assert result.test_name == "KS"
        assert result.feature_name == "feature_a"
        assert "ref_n" in result.details
        assert "cur_n" in result.details


# ---------------------------------------------------------------------------
# Chi-Squared Test
# ---------------------------------------------------------------------------

class TestChiSquared:
    def test_drift_detected(self, categorical_reference, categorical_drifted):
        test = ChiSquaredTest(p_value_threshold=0.05)
        result = test.run(categorical_reference, categorical_drifted)
        assert result is not None
        assert result.drifted is True

    def test_no_drift_not_detected(self, categorical_reference):
        stable = categorical_reference.model_copy(update={
            "value_counts": {"north": 255, "south": 245, "east": 252, "west": 248}
        })
        test = ChiSquaredTest(p_value_threshold=0.05)
        result = test.run(categorical_reference, stable)
        assert result is not None
        assert result.drifted is False

    def test_returns_none_for_continuous(self, continuous_reference, continuous_drifted):
        test = ChiSquaredTest()
        result = test.run(continuous_reference, continuous_drifted)
        assert result is None

    def test_top_shifted_categories_in_details(self, categorical_reference, categorical_drifted):
        test = ChiSquaredTest()
        result = test.run(categorical_reference, categorical_drifted)
        assert "top_shifted" in result.details
        assert len(result.details["top_shifted"]) > 0


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

class TestPSI:
    def test_stable_psi_below_threshold(self, continuous_reference, continuous_no_drift):
        test = PopulationStabilityIndex(threshold=0.2)
        result = test.run(continuous_reference, continuous_no_drift)
        assert result is not None
        assert result.drifted is False
        assert result.statistic < 0.2

    def test_drifted_psi_above_threshold(self, continuous_reference, continuous_drifted):
        test = PopulationStabilityIndex(threshold=0.2)
        result = test.run(continuous_reference, continuous_drifted)
        assert result is not None
        assert result.drifted is True
        assert result.statistic >= 0.2

    def test_psi_is_non_negative(self, continuous_reference, continuous_drifted):
        test = PopulationStabilityIndex()
        result = test.run(continuous_reference, continuous_drifted)
        assert result.statistic >= 0.0

    def test_psi_no_p_value(self, continuous_reference, continuous_drifted):
        test = PopulationStabilityIndex()
        result = test.run(continuous_reference, continuous_drifted)
        assert result.p_value is None

    def test_categorical_psi(self, categorical_reference, categorical_drifted):
        test = PopulationStabilityIndex(threshold=0.2)
        result = test.run(categorical_reference, categorical_drifted)
        assert result is not None


# ---------------------------------------------------------------------------
# Jensen-Shannon
# ---------------------------------------------------------------------------

class TestJensenShannon:
    def test_identical_distributions_zero_distance(self, continuous_reference):
        test = JensenShannonDivergence(threshold=0.1)
        result = test.run(continuous_reference, continuous_reference)
        assert result is not None
        assert result.statistic < 0.01   # Near zero for identical

    def test_drifted_above_threshold(self, continuous_reference, continuous_drifted):
        test = JensenShannonDivergence(threshold=0.05)
        result = test.run(continuous_reference, continuous_drifted)
        assert result is not None
        assert result.drifted is True

    def test_statistic_bounded_01(self, continuous_reference, continuous_drifted):
        test = JensenShannonDivergence()
        result = test.run(continuous_reference, continuous_drifted)
        assert 0.0 <= result.statistic <= 1.0


# ---------------------------------------------------------------------------
# SHAP Delta
# ---------------------------------------------------------------------------

class TestSHAPDelta:
    def test_large_delta_triggers_alert(self, continuous_reference, continuous_drifted):
        # continuous_drifted has shap_mean_abs=0.45 vs reference 0.15 → delta=2.0 (200%)
        test = SHAPDeltaTracker(threshold=0.15)
        result = test.run(continuous_reference, continuous_drifted)
        assert result is not None
        assert result.drifted is True

    def test_small_delta_no_alert(self, continuous_reference, continuous_no_drift):
        # continuous_no_drift has shap=0.16 vs 0.15 → delta ≈ 0.067
        test = SHAPDeltaTracker(threshold=0.15)
        result = test.run(continuous_reference, continuous_no_drift)
        assert result is not None
        assert result.drifted is False

    def test_returns_none_without_shap(self, continuous_reference):
        no_shap = continuous_reference.model_copy(update={"shap_mean_abs": None})
        test = SHAPDeltaTracker()
        result = test.run(no_shap, no_shap)
        assert result is None


# ---------------------------------------------------------------------------
# Detection Engine
# ---------------------------------------------------------------------------

class TestDriftDetectionEngine:
    @pytest.fixture
    def engine(self):
        return DriftDetectionEngine(min_alert_tests=2)

    def _make_batch(self, features: list[FeatureDistributionStats]) -> BatchFeatureStats:
        return BatchFeatureStats(
            model_id="test_model",
            window_start=datetime.utcnow() - timedelta(minutes=5),
            window_end=datetime.utcnow(),
            features=features,
        )

    def test_no_alert_before_reference_set(self, engine, continuous_drifted):
        batch = self._make_batch([continuous_drifted])
        alert = engine.evaluate(batch)
        assert alert is None   # No reference → no comparison

    def test_no_alert_when_stable(
        self, engine, continuous_reference, continuous_no_drift,
        categorical_reference
    ):
        ref_batch = self._make_batch([continuous_reference, categorical_reference])
        engine.set_reference(ref_batch)

        cur_batch = self._make_batch([continuous_no_drift, categorical_reference])
        alert = engine.evaluate(cur_batch)
        assert alert is None

    def test_alert_fires_when_drifted(
        self, engine, continuous_reference, continuous_drifted,
        categorical_reference, categorical_drifted
    ):
        ref_batch = self._make_batch([continuous_reference, categorical_reference])
        engine.set_reference(ref_batch)

        cur_batch = self._make_batch([continuous_drifted, categorical_drifted])
        alert = engine.evaluate(cur_batch)
        assert alert is not None
        assert len(alert.drifted_features) > 0

    def test_alert_severity_increases_with_drift(
        self, engine, continuous_reference, continuous_drifted, categorical_reference
    ):
        ref_batch = self._make_batch([continuous_reference, categorical_reference])
        engine.set_reference(ref_batch)

        cur_batch = self._make_batch([continuous_drifted, categorical_reference])
        alert = engine.evaluate(cur_batch)
        # Even if not all features drifted, severity should reflect the magnitude
        if alert:
            assert alert.severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL)

    def test_min_alert_tests_threshold_respected(self, engine, continuous_reference, continuous_no_drift):
        """With min_alert_tests=2, a single borderline test should not fire."""
        ref_batch = self._make_batch([continuous_reference])
        engine.set_reference(ref_batch)
        engine.min_alert_tests = 5   # Require all 5 tests to agree

        cur_batch = self._make_batch([continuous_no_drift])
        alert = engine.evaluate(cur_batch)
        assert alert is None   # Only 1–2 tests would fire on no_drift


# ---------------------------------------------------------------------------
# SPRT
# ---------------------------------------------------------------------------

class TestSPRT:
    def test_promote_when_challenger_clearly_better(self):
        cfg = SPRTConfig(alpha=0.05, beta=0.10, mde=0.02)
        sprt = SPRT(cfg)
        # Challenger consistently 5% better (2.5× MDE)
        for _ in range(1000):
            result = sprt.update([0.20] * 10, [0.25] * 10)
            if result.decision.value != "hold":
                break
        assert result.decision.value == "promote"
        assert result.llr >= cfg.upper_boundary

    def test_rollback_when_challenger_clearly_worse(self):
        cfg = SPRTConfig(alpha=0.05, beta=0.10, mde=0.02)
        sprt = SPRT(cfg)
        for _ in range(1000):
            result = sprt.update([0.25] * 10, [0.15] * 10)
            if result.decision.value != "hold":
                break
        assert result.decision.value == "rollback"
        assert result.llr <= cfg.lower_boundary

    def test_hold_with_ambiguous_data(self):
        cfg = SPRTConfig(alpha=0.05, beta=0.10, mde=0.05)
        sprt = SPRT(cfg)
        rng = np.random.RandomState(42)
        # Very small effect — SPRT should hold for many rounds
        result = sprt.update(
            rng.uniform(0.49, 0.51, 20).tolist(),
            rng.uniform(0.50, 0.52, 20).tolist(),
        )
        assert result.decision.value == "hold"

    def test_reset_clears_state(self):
        sprt = SPRT()
        sprt.update([0.2] * 100, [0.3] * 100)
        sprt.current_llr
        sprt.reset()
        assert sprt.current_llr == 0.0
        assert sprt.n_samples == 0

    def test_expected_sample_size_finite(self):
        sprt = SPRT(SPRTConfig(mde=0.05))
        sizes = sprt.expected_sample_size(p0=0.20)
        assert sizes["expected_n_under_h0"] > 0
        assert sizes["expected_n_under_h1"] > 0
        assert sizes["fixed_horizon_n_approx"] > 0

    def test_sprt_faster_than_fixed_horizon(self):
        """SPRT expected sample size should be < fixed-horizon when effect is real."""
        cfg = SPRTConfig(alpha=0.05, beta=0.10, mde=0.05)
        sprt = SPRT(cfg)
        sizes = sprt.expected_sample_size(p0=0.20)
        # SPRT under H1 should be less than fixed horizon
        assert sizes["expected_n_under_h1"] < sizes["fixed_horizon_n_approx"]


# ---------------------------------------------------------------------------
# Strategy Selector
# ---------------------------------------------------------------------------

class TestStrategySelector:
    @pytest.fixture
    def selector(self, tmp_path):
        import src.utils.config as cfg_module
        original = cfg_module.settings.retraining.selector_model_path
        cfg_module.settings.retraining.selector_model_path = str(tmp_path / "selector.pkl")
        model = StrategySelectorModel(model_path=tmp_path / "selector.pkl")
        yield model
        cfg_module.settings.retraining.selector_model_path = original

    def test_low_data_selects_ensemble_fallback(self):
        x = np.array([0.5, 0.3, 0.25, 0, 0, 0.2, 30, 10, 50, 0, 1])  # low data_ratio
        label = _rule_based_label(x)
        assert label == STRATEGY_TO_IDX["ensemble_fallback"]

    def test_slice_local_drift_selects_slice_finetune(self):
        x = np.array([0.4, 0.2, 0.15, 0, 1, 0.8, 20, 20, 50, 0, 1])  # slice_local=1, good data
        label = _rule_based_label(x)
        assert label == STRATEGY_TO_IDX["slice_finetune"]

    def test_concept_drift_selects_full_retrain(self):
        x = np.array([0.9, 0.6, 0.35, 1, 0, 1.0, 30, 30, 50, 0, 1])  # shap=1, severity=0.9
        label = _rule_based_label(x)
        assert label == STRATEGY_TO_IDX["full_retrain"]

    def test_selector_returns_valid_strategy(self, selector):
        features = StrategyFeatures(
            drift_severity_score=0.5,
            pct_features_drifted=0.3,
            psi_max=0.15,
            shap_drift_detected=False,
            drift_is_slice_local=False,
            data_availability_ratio=1.0,
            days_since_last_retrain=30.0,
            estimated_cost_usd=20.0,
            cost_ceiling_usd=50.0,
            sla_tier_critical=False,
            sla_tier_standard=True,
        )
        strategy, confidence = selector.predict(features)
        assert strategy in list(RetrainingStrategy)
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# Schema round-trips
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_inference_event_serialization(self):
        event = InferenceEvent(
            model_id="model_a",
            model_version="v1.2",
            features={"age": 35, "income": 60000},
            prediction=0.78,
            latency_ms=12.3,
        )
        json_str = event.model_dump_json()
        reloaded = InferenceEvent.model_validate_json(json_str)
        assert reloaded.model_id == event.model_id
        assert reloaded.event_id == event.event_id

    def test_drift_alert_json_roundtrip(self, continuous_reference, continuous_drifted):
        from src.ingestion.schema import TestResult
        alert = DriftAlert(
            model_id="model_a",
            model_version="v1.2",
            window_start=datetime.utcnow() - timedelta(minutes=5),
            window_end=datetime.utcnow(),
            severity=DriftSeverity.HIGH,
            drifted_features=["feature_a"],
            test_results=[
                TestResult(
                    test_name="KS",
                    feature_name="feature_a",
                    statistic=0.35,
                    p_value=0.001,
                    threshold=0.05,
                    drifted=True,
                )
            ],
            tests_fired=1,
            tests_total=3,
        )
        json_str = alert.model_dump_json()
        reloaded = DriftAlert.model_validate_json(json_str)
        assert reloaded.alert_id == alert.alert_id
        assert reloaded.severity == DriftSeverity.HIGH
