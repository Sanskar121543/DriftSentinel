"""
SHAP Delta Tracker — Concept Drift Detection

Concept drift occurs when the relationship between features and the target
changes, even if the feature distributions themselves are stable.

Standard distribution tests (KS, PSI, JS) CANNOT catch this.

Strategy:
  - Store the mean absolute SHAP value per feature from the reference window
  - Compare to the same metric in the current window
  - A large shift in SHAP importance signals that the model is relying on
    different features than it was trained to, which is concept drift

SHAP values must be computed by the model serving layer and included in
inference events.  We store mean(|SHAP|) per feature per batch via the
feature aggregator (shap_mean_abs column in FeatureDistributionStats).
"""

from __future__ import annotations

import numpy as np

from src.ingestion.schema import FeatureDistributionStats, TestResult

_TEST_NAME = "SHAP_Delta"


class SHAPDeltaTracker:
    """
    Detects concept drift by comparing mean absolute SHAP values between
    reference and current batches.

    Relative delta: |cur - ref| / (ref + eps)
    Alert if delta > threshold for any individual feature, OR if the global
    Euclidean drift of the SHAP importance vector exceeds the threshold.
    """

    def __init__(self, threshold: float = 0.15, use_relative: bool = True) -> None:
        self.threshold = threshold
        self.use_relative = use_relative

    def run(
        self,
        reference: FeatureDistributionStats,
        current: FeatureDistributionStats,
    ) -> TestResult | None:
        ref_shap = reference.shap_mean_abs
        cur_shap = current.shap_mean_abs

        if ref_shap is None or cur_shap is None:
            return None

        eps = 1e-9
        if self.use_relative:
            delta = abs(cur_shap - ref_shap) / (abs(ref_shap) + eps)
        else:
            delta = abs(cur_shap - ref_shap)

        drifted = bool(delta > self.threshold)

        return TestResult(
            test_name=_TEST_NAME,
            feature_name=current.feature_name,
            statistic=float(delta),
            p_value=None,
            threshold=self.threshold,
            drifted=drifted,
            details={
                "ref_shap_mean_abs": float(ref_shap),
                "cur_shap_mean_abs": float(cur_shap),
                "relative_delta": float(delta),
                "metric": "relative" if self.use_relative else "absolute",
                "interpretation": (
                    "Feature importance shifted significantly — likely concept drift"
                    if drifted
                    else "Feature importance stable"
                ),
            },
        )


# ---------------------------------------------------------------------------
# Batch-level SHAP vector drift (for calling from engine with full batch)
# ---------------------------------------------------------------------------

def shap_vector_drift(
    ref_shap_vector: dict[str, float],
    cur_shap_vector: dict[str, float],
) -> dict:
    """
    Compute drift in the full SHAP importance vector across all features.
    Returns cosine distance and L2 distance between importance vectors.
    """
    features = sorted(set(ref_shap_vector) & set(cur_shap_vector))
    if not features:
        return {"cosine_distance": None, "l2_distance": None, "top_shifted": []}

    ref_vec = np.array([ref_shap_vector[f] for f in features])
    cur_vec = np.array([cur_shap_vector[f] for f in features])

    eps = 1e-9
    cosine_dist = 1.0 - float(
        np.dot(ref_vec, cur_vec)
        / (np.linalg.norm(ref_vec) * np.linalg.norm(cur_vec) + eps)
    )
    l2_dist = float(np.linalg.norm(cur_vec - ref_vec))

    deltas = np.abs(cur_vec - ref_vec) / (ref_vec + eps)
    top_idx = np.argsort(deltas)[::-1][:5]
    top_shifted = [
        {
            "feature": features[i],
            "ref": float(ref_vec[i]),
            "cur": float(cur_vec[i]),
            "relative_delta": float(deltas[i]),
        }
        for i in top_idx
    ]

    return {
        "cosine_distance": cosine_dist,
        "l2_distance": l2_dist,
        "top_shifted": top_shifted,
    }
