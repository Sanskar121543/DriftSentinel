"""
Population Stability Index (PSI)

PSI < 0.1   → No significant change
PSI 0.1–0.2 → Moderate change, monitor
PSI > 0.2   → Significant change, action required

Works for both continuous (histogram buckets) and categorical (value counts).
"""

from __future__ import annotations

import numpy as np

from src.ingestion.schema import FeatureDistributionStats, TestResult

_TEST_NAME = "PSI"
_EPSILON = 1e-7   # Prevent log(0)


class PopulationStabilityIndex:
    def __init__(self, threshold: float = 0.2, n_bins: int = 10) -> None:
        self.threshold = threshold
        self.n_bins = n_bins

    def run(
        self,
        reference: FeatureDistributionStats,
        current: FeatureDistributionStats,
    ) -> TestResult | None:
        ref_pct, cur_pct = self._get_distributions(reference, current)
        if ref_pct is None or cur_pct is None:
            return None

        psi = _psi(ref_pct, cur_pct)
        drifted = bool(psi > self.threshold)

        return TestResult(
            test_name=_TEST_NAME,
            feature_name=current.feature_name,
            statistic=float(psi),
            p_value=None,   # PSI has no p-value
            threshold=self.threshold,
            drifted=drifted,
            details={"interpretation": _psi_label(psi)},
        )

    def _get_distributions(
        self,
        ref: FeatureDistributionStats,
        cur: FeatureDistributionStats,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        # Categorical path
        if ref.value_counts and cur.value_counts:
            all_cats = sorted(set(ref.value_counts) | set(cur.value_counts))
            ref_counts = np.array([ref.value_counts.get(c, 0) for c in all_cats], dtype=float)
            cur_counts = np.array([cur.value_counts.get(c, 0) for c in all_cats], dtype=float)
            ref_pct = (ref_counts + _EPSILON) / (ref_counts.sum() + _EPSILON * len(all_cats))
            cur_pct = (cur_counts + _EPSILON) / (cur_counts.sum() + _EPSILON * len(all_cats))
            return ref_pct, cur_pct

        # Continuous path: use histogram if available, else percentile bucketing
        if (
            ref.histogram_edges is not None and ref.histogram_counts is not None
            and cur.histogram_counts is not None
        ):
            ref_counts = np.array(ref.histogram_counts, dtype=float)
            cur_counts = np.array(cur.histogram_counts, dtype=float)
            # Align lengths (current hist may differ in bin count)
            min_len = min(len(ref_counts), len(cur_counts))
            ref_counts = ref_counts[:min_len]
            cur_counts = cur_counts[:min_len]
            ref_pct = (ref_counts + _EPSILON) / (ref_counts.sum() + _EPSILON * min_len)
            cur_pct = (cur_counts + _EPSILON) / (cur_counts.sum() + _EPSILON * min_len)
            return ref_pct, cur_pct

        # Percentile bucketing fallback
        ref_pcts = _percentile_buckets(ref)
        cur_pcts = _percentile_buckets(cur)
        if ref_pcts is None or cur_pcts is None:
            return None, None
        return ref_pcts, cur_pcts


def _percentile_buckets(s: FeatureDistributionStats) -> np.ndarray | None:
    """
    Use stored percentile breakpoints as approximate bucket boundaries.
    Differences between consecutive percentiles ≈ fraction of population in that range.
    """
    pcts = [s.p25, s.p50, s.p75, s.p95, s.p99]
    if any(p is None for p in pcts):
        return None
    # Each percentile gap holds roughly this fraction of the population
    fractions = np.array([0.25, 0.25, 0.25, 0.20, 0.04, 0.01])
    return fractions


def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    return float(np.sum((cur - ref) * np.log((cur + _EPSILON) / (ref + _EPSILON))))


def _psi_label(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    elif psi < 0.2:
        return "moderate_change"
    return "significant_change"
