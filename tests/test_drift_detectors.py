"""
Edge-case and parametrized tests for the 5 drift detectors.

Targets the failure modes that matter in production: empty inputs, single
categories, unseen categories, missing histograms, percentile-only fallback
reconstruction, and the type-routing contract (each test must ignore feature
types it does not handle).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.drift.tests.chi_square import ChiSquaredTest, _top_shifted_categories
from src.drift.tests.jensen_shannon import JensenShannonDivergence
from src.drift.tests.ks_test import KolmogorovSmirnovTest
from src.drift.tests.psi import PopulationStabilityIndex
from src.drift.tests.shap_delta import SHAPDeltaTracker, shap_vector_drift
from src.ingestion.schema import FeatureDistributionStats, FeatureType, TestResult

from tests.conftest import make_categorical, make_continuous


# ---------------------------------------------------------------------------
# Type routing — every detector must return None for the wrong feature type
# ---------------------------------------------------------------------------

class TestTypeRouting:
    def test_ks_skips_categorical(self):
        ref = make_categorical()
        assert KolmogorovSmirnovTest().run(ref, ref) is None

    def test_chi2_skips_continuous(self):
        ref = make_continuous()
        assert ChiSquaredTest().run(ref, ref) is None

    def test_shap_skips_when_missing(self):
        ref = make_continuous(shap=None)
        assert SHAPDeltaTracker().run(ref, ref) is None


# ---------------------------------------------------------------------------
# KS reconstruction
# ---------------------------------------------------------------------------

class TestKSReconstruction:
    def test_percentile_fallback_when_no_histogram(self):
        ref = make_continuous(seed=0)
        cur = make_continuous(loc=3.0, seed=2)
        # Strip histograms → force percentile-based reconstruction path.
        ref = ref.model_copy(update={"histogram_edges": None, "histogram_counts": None})
        cur = cur.model_copy(update={"histogram_edges": None, "histogram_counts": None})
        result = KolmogorovSmirnovTest().run(ref, cur)
        assert result is not None
        assert result.drifted is True

    def test_returns_none_with_insufficient_data(self):
        thin = FeatureDistributionStats(
            model_id="m",
            feature_name="f",
            feature_type=FeatureType.CONTINUOUS,
            window_start=datetime.utcnow() - timedelta(minutes=5),
            window_end=datetime.utcnow(),
            histogram_edges=[0.0, 1.0],
            histogram_counts=[2],
            total_count=2,
        )
        assert KolmogorovSmirnovTest().run(thin, thin) is None

    def test_details_carry_sample_counts(self):
        ref = make_continuous(seed=0)
        cur = make_continuous(loc=2.0, seed=2, edges=ref.histogram_edges)
        result = KolmogorovSmirnovTest().run(ref, cur)
        assert result.details["ref_n"] > 0
        assert result.details["cur_n"] > 0


# ---------------------------------------------------------------------------
# Chi-Squared
# ---------------------------------------------------------------------------

class TestChiSquaredEdges:
    def test_unseen_category_is_aligned(self):
        ref = make_categorical(value_counts={"a": 100, "b": 100})
        cur = make_categorical(value_counts={"a": 100, "b": 100, "c": 100})
        result = ChiSquaredTest().run(ref, cur)
        assert result is not None
        assert result.details["categories"] == 3

    def test_empty_value_counts_returns_none(self):
        ref = make_categorical(value_counts={"a": 1})
        empty = ref.model_copy(update={"value_counts": {}})
        assert ChiSquaredTest().run(ref, empty) is None

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1])
    def test_threshold_recorded_on_result(self, threshold):
        ref = make_categorical()
        cur = make_categorical(value_counts={"north": 700, "south": 100, "east": 100, "west": 100})
        result = ChiSquaredTest(p_value_threshold=threshold).run(ref, cur)
        assert result.threshold == threshold

    def test_top_shifted_helper_ranks_by_delta(self):
        cats = ["a", "b", "c"]
        ref = np.array([100.0, 100.0, 100.0])
        cur = np.array([300.0, 100.0, 100.0])
        top = _top_shifted_categories(ref, cur, cats, top_n=1)
        assert top[0]["category"] == "a"


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

class TestPSIEdges:
    def test_interpretation_label_present(self):
        ref = make_continuous(seed=0)
        cur = make_continuous(loc=2.0, seed=2, edges=ref.histogram_edges)
        result = PopulationStabilityIndex().run(ref, cur)
        assert result.details["interpretation"] in (
            "stable",
            "moderate_change",
            "significant_change",
        )

    def test_psi_has_no_pvalue(self):
        ref = make_continuous(seed=0)
        result = PopulationStabilityIndex().run(ref, ref)
        assert result.p_value is None

    def test_categorical_psi_runs(self):
        ref = make_categorical()
        cur = make_categorical(value_counts={"north": 600, "south": 100, "east": 200, "west": 100})
        result = PopulationStabilityIndex().run(ref, cur)
        assert result is not None
        assert result.statistic >= 0.0


# ---------------------------------------------------------------------------
# Jensen-Shannon
# ---------------------------------------------------------------------------

class TestJensenShannonEdges:
    def test_divergence_is_squared_distance(self):
        ref = make_continuous(seed=0)
        cur = make_continuous(loc=2.0, seed=2, edges=ref.histogram_edges)
        result = JensenShannonDivergence().run(ref, cur)
        assert result.details["js_divergence"] == pytest.approx(result.statistic ** 2, rel=1e-6)

    def test_categorical_distance_symmetric(self):
        a = make_categorical(value_counts={"x": 100, "y": 50})
        b = make_categorical(value_counts={"x": 50, "y": 100})
        ab = JensenShannonDivergence().run(a, b).statistic
        ba = JensenShannonDivergence().run(b, a).statistic
        assert ab == pytest.approx(ba, rel=1e-6)


# ---------------------------------------------------------------------------
# SHAP delta + vector drift
# ---------------------------------------------------------------------------

class TestSHAPDelta:
    def test_absolute_mode(self):
        ref = make_continuous(shap=0.20, seed=0)
        cur = make_continuous(shap=0.30, seed=1, edges=ref.histogram_edges)
        result = SHAPDeltaTracker(threshold=0.05, use_relative=False).run(ref, cur)
        assert result.details["metric"] == "absolute"
        assert result.statistic == pytest.approx(0.10, abs=1e-6)

    def test_relative_delta_value(self):
        ref = make_continuous(shap=0.20, seed=0)
        cur = make_continuous(shap=0.40, seed=1, edges=ref.histogram_edges)
        result = SHAPDeltaTracker(threshold=0.15).run(ref, cur)
        assert result.statistic == pytest.approx(1.0, rel=1e-6)
        assert result.drifted is True

    def test_vector_drift_cosine_and_l2(self):
        out = shap_vector_drift({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0})
        assert out["cosine_distance"] == pytest.approx(1.0, abs=1e-6)
        assert out["l2_distance"] == pytest.approx(np.sqrt(2), rel=1e-6)

    def test_vector_drift_no_shared_features(self):
        out = shap_vector_drift({"a": 1.0}, {"b": 1.0})
        assert out["cosine_distance"] is None
        assert out["top_shifted"] == []


# ---------------------------------------------------------------------------
# Shared TestResult contract
# ---------------------------------------------------------------------------

def test_all_detectors_emit_well_formed_results():
    ref_c = make_continuous(seed=0)
    cur_c = make_continuous(loc=2.0, seed=2, edges=ref_c.histogram_edges)
    ref_cat = make_categorical()
    cur_cat = make_categorical(value_counts={"north": 600, "south": 100, "east": 200, "west": 100})

    results = [
        KolmogorovSmirnovTest().run(ref_c, cur_c),
        PopulationStabilityIndex().run(ref_c, cur_c),
        JensenShannonDivergence().run(ref_c, cur_c),
        SHAPDeltaTracker().run(ref_c, cur_c),
        ChiSquaredTest().run(ref_cat, cur_cat),
    ]
    for r in results:
        assert isinstance(r, TestResult)
        assert isinstance(r.drifted, bool)
        assert r.feature_name
        assert r.threshold is not None
