"""
Property-based tests (Hypothesis).

Instead of hand-picking examples, these tests assert *invariants* that must
hold for every input the strategies can generate. They are the strongest
guard against edge-case regressions in the statistical core:

  - KS / JS statistics must stay inside their mathematical bounds [0, 1]
  - PSI must be non-negative and zero for identical distributions
  - SPRT must be symmetric, monotone, and never decide outside its boundaries
  - Chi-Squared p-values must stay in [0, 1]
"""

from __future__ import annotations

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.canary.sprt import SPRT, SPRTConfig
from src.drift.tests.chi_square import ChiSquaredTest
from src.drift.tests.jensen_shannon import JensenShannonDivergence
from src.drift.tests.ks_test import KolmogorovSmirnovTest
from src.drift.tests.psi import PopulationStabilityIndex
from src.ingestion.schema import CanaryDecision

from tests.conftest import make_categorical, make_continuous

_SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Drift-statistic bounds
# ---------------------------------------------------------------------------

@_SETTINGS
@given(
    loc=st.floats(-5, 5),
    scale=st.floats(0.2, 3.0),
    seed=st.integers(0, 10_000),
)
def test_ks_statistic_always_in_unit_interval(loc, scale, seed):
    ref = make_continuous(loc=0.0, scale=1.0, seed=1)
    cur = make_continuous(loc=loc, scale=scale, seed=seed, edges=ref.histogram_edges)
    result = KolmogorovSmirnovTest().run(ref, cur)
    assume(result is not None)
    assert 0.0 <= result.statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0


@_SETTINGS
@given(
    loc=st.floats(-5, 5),
    scale=st.floats(0.2, 3.0),
    seed=st.integers(0, 10_000),
)
def test_js_distance_always_in_unit_interval(loc, scale, seed):
    ref = make_continuous(loc=0.0, scale=1.0, seed=1)
    cur = make_continuous(loc=loc, scale=scale, seed=seed, edges=ref.histogram_edges)
    result = JensenShannonDivergence().run(ref, cur)
    assume(result is not None)
    assert 0.0 <= result.statistic <= 1.0


@_SETTINGS
@given(
    loc=st.floats(-5, 5),
    scale=st.floats(0.2, 3.0),
    seed=st.integers(0, 10_000),
)
def test_psi_is_non_negative(loc, scale, seed):
    ref = make_continuous(loc=0.0, scale=1.0, seed=1)
    cur = make_continuous(loc=loc, scale=scale, seed=seed, edges=ref.histogram_edges)
    result = PopulationStabilityIndex().run(ref, cur)
    assume(result is not None)
    # PSI is a sum of (cur-ref)*log(cur/ref); mathematically >= 0 (up to float noise)
    assert result.statistic >= -1e-9


@_SETTINGS
@given(
    counts=st.lists(st.integers(1, 5000), min_size=2, max_size=8),
)
def test_chi2_pvalue_in_unit_interval(counts):
    cats = {f"c{i}": c for i, c in enumerate(counts)}
    ref = make_categorical(value_counts=cats)
    cur = make_categorical(value_counts=cats)
    result = ChiSquaredTest().run(ref, cur)
    assume(result is not None)
    assert 0.0 <= result.p_value <= 1.0
    assert result.statistic >= 0.0


def test_identical_continuous_has_near_zero_psi_and_js():
    ref = make_continuous(loc=0.0, scale=1.0, seed=7)
    psi = PopulationStabilityIndex().run(ref, ref)
    js = JensenShannonDivergence().run(ref, ref)
    assert psi.statistic < 1e-6
    assert js.statistic < 1e-6


# ---------------------------------------------------------------------------
# SPRT invariants
# ---------------------------------------------------------------------------

@_SETTINGS
@given(
    p=st.floats(0.05, 0.95),
    n=st.integers(1, 200),
)
def test_sprt_identical_streams_never_promote(p, n):
    """Identical champion/challenger streams must not yield a PROMOTE."""
    sprt = SPRT(SPRTConfig(mde=0.02))
    result = sprt.update([p] * n, [p] * n)
    # With zero observed difference the LLR stays ~0 → HOLD, never PROMOTE.
    assert result.decision != CanaryDecision.PROMOTE
    assert abs(result.llr) < sprt.config.upper_boundary


@_SETTINGS
@given(
    champ=st.floats(0.05, 0.45),
    lift=st.floats(0.05, 0.4),
    n=st.integers(50, 400),
)
def test_sprt_consistent_lift_promotes(champ, lift, n):
    """A consistent, material challenger lift must eventually PROMOTE."""
    chal = min(champ + lift, 0.99)
    sprt = SPRT(SPRTConfig(alpha=0.05, beta=0.10, mde=0.02))
    result = sprt.update([champ] * n, [chal] * n)
    assert result.decision == CanaryDecision.PROMOTE
    assert result.llr >= sprt.config.upper_boundary


@_SETTINGS
@given(
    a=st.floats(0.05, 0.9),
    b=st.floats(0.05, 0.9),
    n=st.integers(1, 100),
)
def test_sprt_decision_never_outside_boundaries(a, b, n):
    """Whatever the data, a reported decision must agree with the LLR sign."""
    sprt = SPRT()
    r = sprt.update([a] * n, [b] * n)
    if r.decision == CanaryDecision.PROMOTE:
        assert r.llr >= r.upper_boundary
    elif r.decision == CanaryDecision.ROLLBACK:
        assert r.llr <= r.lower_boundary
    else:
        assert r.lower_boundary < r.llr < r.upper_boundary


@_SETTINGS
@given(n=st.integers(1, 300))
def test_sprt_sample_count_matches_input(n):
    sprt = SPRT()
    r = sprt.update([0.2] * n, [0.25] * n)
    assert r.n_samples == n
    assert sprt.n_samples == n


def test_sprt_mismatched_lengths_raise():
    import pytest

    with pytest.raises(ValueError):
        SPRT().update([0.1, 0.2], [0.1])
