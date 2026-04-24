"""Chi-Squared drift test for categorical features."""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.ingestion.schema import FeatureDistributionStats, TestResult

_TEST_NAME = "Chi2"


class ChiSquaredTest:
    def __init__(self, p_value_threshold: float = 0.05) -> None:
        self.p_value_threshold = p_value_threshold

    def run(
        self,
        reference: FeatureDistributionStats,
        current: FeatureDistributionStats,
    ) -> TestResult | None:
        if not reference.value_counts or not current.value_counts:
            return None

        # Align categories
        all_cats = sorted(set(reference.value_counts) | set(current.value_counts))
        ref_counts = np.array([reference.value_counts.get(c, 0) for c in all_cats], dtype=float)
        cur_counts = np.array([current.value_counts.get(c, 0) for c in all_cats], dtype=float)

        if ref_counts.sum() == 0 or cur_counts.sum() == 0:
            return None

        # Scale reference to same total as current (chi2 requires expected counts)
        expected = ref_counts / ref_counts.sum() * cur_counts.sum()
        # Add small epsilon to avoid zero-expected cells
        expected = np.maximum(expected, 1e-6)

        statistic, p_value = stats.chisquare(f_obs=cur_counts, f_exp=expected)
        drifted = p_value < self.p_value_threshold

        return TestResult(
            test_name=_TEST_NAME,
            feature_name=current.feature_name,
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=self.p_value_threshold,
            drifted=drifted,
            details={
                "categories": len(all_cats),
                "top_shifted": _top_shifted_categories(ref_counts, cur_counts, all_cats),
            },
        )


def _top_shifted_categories(
    ref: np.ndarray, cur: np.ndarray, cats: list[str], top_n: int = 3
) -> list[dict]:
    ref_pct = ref / ref.sum()
    cur_pct = cur / cur.sum()
    delta = np.abs(cur_pct - ref_pct)
    top_idx = np.argsort(delta)[::-1][:top_n]
    return [
        {
            "category": cats[i],
            "ref_pct": float(ref_pct[i]),
            "cur_pct": float(cur_pct[i]),
            "delta": float(delta[i]),
        }
        for i in top_idx
    ]
