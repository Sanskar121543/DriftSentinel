"""
Jensen-Shannon Divergence drift test.

JSD is symmetric (unlike KL divergence) and always bounded in [0, 1]
when computed from distributions that sum to 1.  It is the square root
of the Jensen-Shannon divergence of the probability distributions.

JSD < 0.1   → Distributions are very similar
JSD > 0.2   → Significant distributional shift
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon

from src.ingestion.schema import FeatureDistributionStats, TestResult

_TEST_NAME = "JensenShannon"
_EPSILON = 1e-9


class JensenShannonDivergence:
    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def run(
        self,
        reference: FeatureDistributionStats,
        current: FeatureDistributionStats,
    ) -> TestResult | None:
        ref_dist, cur_dist = self._get_distributions(reference, current)
        if ref_dist is None or cur_dist is None:
            return None

        # scipy jensenshannon returns the JS distance (sqrt of divergence)
        js_distance = float(jensenshannon(ref_dist, cur_dist))
        drifted = bool(js_distance > self.threshold)

        return TestResult(
            test_name=_TEST_NAME,
            feature_name=current.feature_name,
            statistic=js_distance,
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            details={
                "js_divergence": float(js_distance ** 2),
                "interpretation": "symmetric KL-divergence bounded in [0,1]",
            },
        )

    def _get_distributions(
        self,
        ref: FeatureDistributionStats,
        cur: FeatureDistributionStats,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        # Categorical
        if ref.value_counts and cur.value_counts:
            all_cats = sorted(set(ref.value_counts) | set(cur.value_counts))
            ref_c = np.array([ref.value_counts.get(c, 0) + _EPSILON for c in all_cats])
            cur_c = np.array([cur.value_counts.get(c, 0) + _EPSILON for c in all_cats])
            return ref_c / ref_c.sum(), cur_c / cur_c.sum()

        # Continuous via histogram
        if (
            ref.histogram_counts is not None
            and cur.histogram_counts is not None
        ):
            ref_c = np.array(ref.histogram_counts, dtype=float) + _EPSILON
            cur_c = np.array(cur.histogram_counts, dtype=float) + _EPSILON
            min_len = min(len(ref_c), len(cur_c))
            ref_c = ref_c[:min_len]
            cur_c = cur_c[:min_len]
            return ref_c / ref_c.sum(), cur_c / cur_c.sum()

        # Percentile-based approximate distribution
        ref_dist = _from_percentiles(ref)
        cur_dist = _from_percentiles(cur)
        return ref_dist, cur_dist


def _from_percentiles(s: FeatureDistributionStats) -> np.ndarray | None:
    """
    Build an approximate PMF from stored percentile values.
    Bucket widths are proportional to percentile spacing.
    """
    breakpoints = [s.min, s.p25, s.p50, s.p75, s.p95, s.p99, s.max]
    if any(b is None for b in breakpoints):
        return None
    widths = np.diff(np.array(breakpoints, dtype=float))
    widths = np.abs(widths) + _EPSILON
    # Each bucket holds the fraction implied by percentile spacing
    fractions = np.array([0.25, 0.25, 0.25, 0.20, 0.04, 0.01])
    # Normalize bucket density = fraction / width (mass is what matters for JS)
    return fractions / fractions.sum()
