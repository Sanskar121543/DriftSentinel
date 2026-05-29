"""
Kolmogorov-Smirnov Test for continuous feature drift.

Reconstructs empirical distributions from stored histogram data (edges +
counts) and computes the KS statistic between reference and current.

Using scipy.stats.ks_2samp when raw samples are available,
and a histogram-based approximation when only binned data is stored.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.ingestion.schema import FeatureDistributionStats, TestResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

_TEST_NAME = "KS"


class KolmogorovSmirnovTest:
    def __init__(self, p_value_threshold: float = 0.05) -> None:
        self.p_value_threshold = p_value_threshold

    def run(
        self,
        reference: FeatureDistributionStats,
        current: FeatureDistributionStats,
    ) -> TestResult | None:
        if reference.feature_type.value not in ("continuous",):
            return None

        ref_samples = self._reconstruct_samples(reference)
        cur_samples = self._reconstruct_samples(current)

        if ref_samples is None or cur_samples is None or len(ref_samples) < 5 or len(cur_samples) < 5:
            logger.debug(
                "ks_insufficient_data",
                feature=current.feature_name,
                ref_n=len(ref_samples) if ref_samples is not None else 0,
                cur_n=len(cur_samples) if cur_samples is not None else 0,
            )
            return None

        statistic, p_value = stats.ks_2samp(ref_samples, cur_samples)
        drifted = bool(p_value < self.p_value_threshold)

        return TestResult(
            test_name=_TEST_NAME,
            feature_name=current.feature_name,
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=self.p_value_threshold,
            drifted=drifted,
            details={
                "ref_n": len(ref_samples),
                "cur_n": len(cur_samples),
                "ref_mean": float(np.mean(ref_samples)),
                "cur_mean": float(np.mean(cur_samples)),
            },
        )

    def _reconstruct_samples(
        self, stats_obj: FeatureDistributionStats
    ) -> np.ndarray | None:
        """
        Reconstruct approximate sample array from stored histogram.
        Each bin is represented by `count` copies of its midpoint.
        Falls back to percentile-based reconstruction if no histogram.
        """
        if (
            stats_obj.histogram_edges is not None
            and stats_obj.histogram_counts is not None
            and len(stats_obj.histogram_edges) > 1
        ):
            edges = np.array(stats_obj.histogram_edges)
            counts = np.array(stats_obj.histogram_counts, dtype=int)
            midpoints = (edges[:-1] + edges[1:]) / 2
            samples = np.repeat(midpoints, counts)
            return samples

        # Fallback: reconstruct from stored percentiles using linear interpolation
        pcts = {
            0.0: stats_obj.min,
            0.25: stats_obj.p25,
            0.50: stats_obj.p50,
            0.75: stats_obj.p75,
            0.95: stats_obj.p95,
            0.99: stats_obj.p99,
            1.0: stats_obj.max,
        }
        valid = [(q, v) for q, v in pcts.items() if v is not None]
        if len(valid) < 3:
            return None

        quantiles, values = zip(*sorted(valid))
        n = max(stats_obj.total_count, 100)
        interp_quantiles = np.linspace(0, 1, n)
        samples = np.interp(interp_quantiles, quantiles, values)
        return samples
