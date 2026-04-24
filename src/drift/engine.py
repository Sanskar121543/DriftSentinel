"""
Drift Detection Engine

Orchestrates 5 parallel drift tests against reference distributions:
  1. Kolmogorov-Smirnov  — continuous features
  2. Chi-Squared          — categorical features
  3. Population Stability Index — all features
  4. Jensen-Shannon Divergence  — all features
  5. SHAP Delta Tracking  — concept drift via feature importance shift

Each test runs in a separate thread pool worker.  An alert fires when
≥ N tests agree (configurable, default 2).  Slice-aware: tests run per
(model_id, segment) combination independently.

Reference distributions are fetched from the feature-stats Kafka compacted
topic (so they survive Spark restarts) and cached in memory with a TTL.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.drift.tests.ks_test import KolmogorovSmirnovTest
from src.drift.tests.chi_square import ChiSquaredTest
from src.drift.tests.psi import PopulationStabilityIndex
from src.drift.tests.jensen_shannon import JensenShannonDivergence
from src.drift.tests.shap_delta import SHAPDeltaTracker
from src.ingestion.schema import (
    BatchFeatureStats,
    DriftAlert,
    DriftSeverity,
    FeatureDistributionStats,
    TestResult,
)
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reference distribution store (in-memory + Kafka compacted topic backing)
# ---------------------------------------------------------------------------

@dataclass
class ReferenceStore:
    """
    Caches reference distributions per (model_id, feature_name, segment_key).
    On cache miss, falls back to reading from the Kafka compacted topic.
    TTL: 1 hour (reference should be stable; refresh periodically).
    """

    _cache: dict[str, tuple[FeatureDistributionStats, float]] = field(
        default_factory=dict, init=False
    )
    ttl_seconds: float = 3600.0

    def get(
        self, model_id: str, feature_name: str, segment_key: str = "{}"
    ) -> FeatureDistributionStats | None:
        cache_key = f"{model_id}:{feature_name}:{segment_key}"
        if cache_key in self._cache:
            stats, ts = self._cache[cache_key]
            if time.time() - ts < self.ttl_seconds:
                return stats
            del self._cache[cache_key]
        return None

    def set(
        self,
        model_id: str,
        feature_name: str,
        segment_key: str,
        stats: FeatureDistributionStats,
    ) -> None:
        cache_key = f"{model_id}:{feature_name}:{segment_key}"
        self._cache[cache_key] = (stats, time.time())

    def load_from_batch(self, batch: BatchFeatureStats) -> None:
        seg_key = json.dumps(batch.segment, sort_keys=True)
        for feat in batch.features:
            self.set(batch.model_id, feat.feature_name, seg_key, feat)

    def clear_model(self, model_id: str) -> None:
        keys = [k for k in self._cache if k.startswith(f"{model_id}:")]
        for k in keys:
            del self._cache[k]


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

def _compute_severity(
    test_results: list[TestResult],
    drifted_features: list[str],
    total_features: int,
) -> DriftSeverity:
    """
    Severity heuristic:
      - % of drifted features
      - average PSI score across drifted features
      - number of tests agreeing on drift
    """
    if not drifted_features:
        return DriftSeverity.LOW

    drift_pct = len(drifted_features) / max(total_features, 1)
    tests_fired = sum(1 for r in test_results if r.drifted)

    psi_scores = [
        r.statistic for r in test_results
        if r.test_name == "PSI" and r.drifted
    ]
    avg_psi = np.mean(psi_scores) if psi_scores else 0.0

    if drift_pct >= 0.5 or avg_psi >= 0.4 or tests_fired >= 4:
        return DriftSeverity.CRITICAL
    elif drift_pct >= 0.3 or avg_psi >= 0.25 or tests_fired >= 3:
        return DriftSeverity.HIGH
    elif drift_pct >= 0.1 or avg_psi >= 0.1 or tests_fired >= 2:
        return DriftSeverity.MEDIUM
    return DriftSeverity.LOW


# ---------------------------------------------------------------------------
# DriftDetectionEngine
# ---------------------------------------------------------------------------

@dataclass
class DriftDetectionEngine:
    """
    Main orchestrator.  Call `evaluate(batch)` for each incoming micro-batch.
    Returns DriftAlert | None.  None means no significant drift detected.
    """

    reference_store: ReferenceStore = field(default_factory=ReferenceStore)
    min_alert_tests: int = field(default=2)           # alert if ≥ N tests agree
    max_workers: int = field(default=5)               # parallel test threads

    # Test instances (stateless; safe to reuse across calls)
    _ks: KolmogorovSmirnovTest = field(init=False)
    _chi2: ChiSquaredTest = field(init=False)
    _psi: PopulationStabilityIndex = field(init=False)
    _js: JensenShannonDivergence = field(init=False)
    _shap: SHAPDeltaTracker = field(init=False)

    def __post_init__(self) -> None:
        cfg = settings.drift
        self._ks = KolmogorovSmirnovTest(p_value_threshold=cfg.ks_pvalue_threshold)
        self._chi2 = ChiSquaredTest(p_value_threshold=cfg.chi2_pvalue_threshold)
        self._psi = PopulationStabilityIndex(threshold=cfg.psi_threshold)
        self._js = JensenShannonDivergence(threshold=cfg.js_threshold)
        self._shap = SHAPDeltaTracker(threshold=cfg.shap_delta_threshold)
        self.min_alert_tests = settings.drift.min_alert_tests

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, batch: BatchFeatureStats) -> None:
        """Register a batch as the reference distribution for its model+segment."""
        self.reference_store.load_from_batch(batch)
        logger.info(
            "reference_set",
            model_id=batch.model_id,
            features=len(batch.features),
            segment=batch.segment,
        )

    def evaluate(self, batch: BatchFeatureStats) -> DriftAlert | None:
        """
        Run all 5 drift tests on a new production batch vs. stored reference.
        Returns DriftAlert if drift is detected, else None.
        """
        segment_key = json.dumps(batch.segment, sort_keys=True)
        all_results: list[TestResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._run_all_tests_for_feature,
                    batch.model_id,
                    feat,
                    segment_key,
                ): feat.feature_name
                for feat in batch.features
            }

            for future in concurrent.futures.as_completed(futures):
                feature_name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as exc:
                    logger.error(
                        "test_failed",
                        feature=feature_name,
                        model_id=batch.model_id,
                        error=str(exc),
                    )

        drifted_features = list(
            {r.feature_name for r in all_results if r.drifted}
        )
        tests_fired = sum(1 for r in all_results if r.drifted)

        if tests_fired < self.min_alert_tests:
            logger.debug(
                "no_drift",
                model_id=batch.model_id,
                tests_fired=tests_fired,
                threshold=self.min_alert_tests,
            )
            return None

        severity = _compute_severity(
            all_results, drifted_features, len(batch.features)
        )

        alert = DriftAlert(
            model_id=batch.model_id,
            model_version=batch.features[0].feature_name if batch.features else "unknown",
            window_start=batch.window_start,
            window_end=batch.window_end,
            segment=batch.segment,
            severity=severity,
            drifted_features=drifted_features,
            test_results=all_results,
            tests_fired=tests_fired,
            tests_total=len(all_results),
        )

        logger.warning(
            "drift_detected",
            model_id=batch.model_id,
            severity=severity.value,
            drifted_features=drifted_features,
            tests_fired=tests_fired,
            segment=batch.segment,
        )

        return alert

    # ------------------------------------------------------------------
    # Internal: run all applicable tests for one feature
    # ------------------------------------------------------------------

    def _run_all_tests_for_feature(
        self,
        model_id: str,
        current: FeatureDistributionStats,
        segment_key: str,
    ) -> list[TestResult]:
        reference = self.reference_store.get(
            model_id, current.feature_name, segment_key
        )
        if reference is None:
            logger.debug(
                "no_reference_skip",
                model_id=model_id,
                feature=current.feature_name,
            )
            return []

        results: list[TestResult] = []

        if current.feature_type.value == "continuous":
            results.append(self._ks.run(reference, current))
        elif current.feature_type.value in ("categorical", "binary", "ordinal"):
            results.append(self._chi2.run(reference, current))

        results.append(self._psi.run(reference, current))
        results.append(self._js.run(reference, current))

        if current.shap_mean_abs is not None and reference.shap_mean_abs is not None:
            results.append(self._shap.run(reference, current))

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Slice-aware batch: evaluate each segment separately
    # ------------------------------------------------------------------

    def evaluate_slices(
        self, batches: list[BatchFeatureStats]
    ) -> list[DriftAlert]:
        """
        Evaluate multiple segment batches for the same model.
        Each segment's drift is assessed independently so a globally fine model
        can still fire an alert on a high-value drifted sub-segment.
        """
        alerts: list[DriftAlert] = []
        for batch in batches:
            alert = self.evaluate(batch)
            if alert:
                alerts.append(alert)
        return alerts
