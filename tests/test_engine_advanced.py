"""
Detection-engine and reference-store integration tests.

Covers orchestration behavior that the per-detector tests cannot:
reference TTL expiry, segment-keyed isolation, severity escalation,
slice-aware multi-batch evaluation, and resilience to a detector that
raises inside the thread pool.
"""

from __future__ import annotations

import time

import pytest

from src.drift.engine import (
    DriftDetectionEngine,
    ReferenceStore,
    _compute_severity,
)
from src.ingestion.schema import DriftSeverity, TestResult

from tests.conftest import make_batch, make_continuous


# ---------------------------------------------------------------------------
# ReferenceStore
# ---------------------------------------------------------------------------

class TestReferenceStore:
    def test_set_and_get_roundtrip(self):
        store = ReferenceStore()
        feat = make_continuous()
        store.set("m", "f", "{}", feat)
        assert store.get("m", "f", "{}") is feat

    def test_ttl_expiry_evicts(self):
        store = ReferenceStore(ttl_seconds=0.05)
        store.set("m", "f", "{}", make_continuous())
        time.sleep(0.1)
        assert store.get("m", "f", "{}") is None

    def test_segment_isolation(self):
        store = ReferenceStore()
        store.set("m", "f", '{"region": "us"}', make_continuous(name="f"))
        assert store.get("m", "f", "{}") is None
        assert store.get("m", "f", '{"region": "us"}') is not None

    def test_clear_model_scoped(self):
        store = ReferenceStore()
        store.set("m1", "f", "{}", make_continuous())
        store.set("m2", "f", "{}", make_continuous())
        store.clear_model("m1")
        assert store.get("m1", "f", "{}") is None
        assert store.get("m2", "f", "{}") is not None

    def test_load_from_batch_registers_all_features(self):
        store = ReferenceStore()
        batch = make_batch([make_continuous(name="a"), make_continuous(name="b")])
        store.load_from_batch(batch)
        assert store.get("test_model", "a", "{}") is not None
        assert store.get("test_model", "b", "{}") is not None


# ---------------------------------------------------------------------------
# Severity heuristic
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_no_drift_is_low(self):
        assert _compute_severity([], [], 4) == DriftSeverity.LOW

    def test_high_psi_drives_critical(self):
        results = [
            TestResult(
                test_name="PSI",
                feature_name="f",
                statistic=0.5,
                threshold=0.2,
                drifted=True,
            )
        ]
        assert _compute_severity(results, ["f"], 4) == DriftSeverity.CRITICAL

    def test_many_features_drives_critical(self):
        feats = ["a", "b", "c"]
        results = [
            TestResult(test_name="KS", feature_name=f, statistic=0.3, threshold=0.05, drifted=True)
            for f in feats
        ]
        assert _compute_severity(results, feats, 4) == DriftSeverity.CRITICAL

    def test_single_feature_moderate(self):
        results = [
            TestResult(test_name="KS", feature_name="a", statistic=0.3, threshold=0.05, drifted=True),
            TestResult(test_name="PSI", feature_name="a", statistic=0.15, threshold=0.2, drifted=True),
        ]
        sev = _compute_severity(results, ["a"], 4)
        assert sev in (DriftSeverity.MEDIUM, DriftSeverity.HIGH)


# ---------------------------------------------------------------------------
# Engine orchestration
# ---------------------------------------------------------------------------

class TestEngine:
    @pytest.fixture
    def engine(self):
        eng = DriftDetectionEngine()
        eng.min_alert_tests = 2
        return eng

    def test_no_reference_no_alert(self, engine, continuous_drifted):
        assert engine.evaluate(make_batch([continuous_drifted])) is None

    def test_stable_batch_no_alert(self, engine, continuous_reference, continuous_no_drift):
        engine.set_reference(make_batch([continuous_reference]))
        assert engine.evaluate(make_batch([continuous_no_drift])) is None

    def test_drift_fires_alert(self, engine, continuous_reference, continuous_drifted):
        engine.set_reference(make_batch([continuous_reference]))
        alert = engine.evaluate(make_batch([continuous_drifted]))
        assert alert is not None
        assert "feature_a" in alert.drifted_features
        assert alert.tests_fired >= 2

    def test_high_threshold_suppresses_alert(self, engine, continuous_reference, continuous_drifted):
        engine.set_reference(make_batch([continuous_reference]))
        engine.min_alert_tests = 99
        assert engine.evaluate(make_batch([continuous_drifted])) is None

    def test_detector_exception_is_isolated(self, engine, continuous_reference, continuous_drifted, monkeypatch):
        """A detector raising mid-batch must not crash the whole evaluation."""
        engine.set_reference(make_batch([continuous_reference]))

        def boom(*_a, **_k):
            raise RuntimeError("simulated detector failure")

        monkeypatch.setattr(engine._ks, "run", boom)
        # Other detectors still run; evaluate should not raise.
        engine.evaluate(make_batch([continuous_drifted]))

    def test_evaluate_slices_independent(self, engine, continuous_reference, continuous_drifted):
        engine.set_reference(make_batch([continuous_reference]))
        stable_batch = make_batch([continuous_reference])
        drift_batch = make_batch([continuous_drifted])
        alerts = engine.evaluate_slices([stable_batch, drift_batch])
        assert len(alerts) == 1

    def test_alert_severity_is_enum(self, engine, continuous_reference, continuous_drifted, categorical_reference, categorical_drifted):
        engine.set_reference(make_batch([continuous_reference, categorical_reference]))
        alert = engine.evaluate(make_batch([continuous_drifted, categorical_drifted]))
        assert alert is not None
        assert alert.severity in list(DriftSeverity)
