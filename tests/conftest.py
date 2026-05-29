"""
Shared pytest fixtures and Hypothesis strategies for the DriftSentinel suite.

Centralizes the synthetic-distribution builders used across the unit,
property-based, and integration tests so individual test modules stay focused
on behavior rather than fixture plumbing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.ingestion.schema import (
    BatchFeatureStats,
    FeatureDistributionStats,
    FeatureType,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_continuous(
    *,
    name: str = "feature_a",
    loc: float = 0.0,
    scale: float = 1.0,
    n: int = 1000,
    seed: int = 0,
    edges: list[float] | None = None,
    shap: float | None = 0.15,
    model_id: str = "test_model",
) -> FeatureDistributionStats:
    """Build a continuous feature-stats object from a normal sample."""
    rng = np.random.RandomState(seed)
    data = rng.normal(loc, scale, n)
    if edges is None:
        edges = np.linspace(data.min(), data.max(), 21).tolist()
    counts, _ = np.histogram(data, bins=edges)
    return FeatureDistributionStats(
        model_id=model_id,
        feature_name=name,
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
        total_count=n,
        null_count=0,
        shap_mean_abs=shap,
    )


def make_categorical(
    *,
    name: str = "region",
    value_counts: dict[str, int] | None = None,
    model_id: str = "test_model",
) -> FeatureDistributionStats:
    """Build a categorical feature-stats object."""
    if value_counts is None:
        value_counts = {"north": 250, "south": 250, "east": 250, "west": 250}
    return FeatureDistributionStats(
        model_id=model_id,
        feature_name=name,
        feature_type=FeatureType.CATEGORICAL,
        window_start=datetime.utcnow() - timedelta(minutes=10),
        window_end=datetime.utcnow() - timedelta(minutes=5),
        value_counts=dict(value_counts),
        cardinality=len(value_counts),
        total_count=sum(value_counts.values()),
        null_count=0,
    )


def make_batch(
    features: list[FeatureDistributionStats], model_id: str = "test_model"
) -> BatchFeatureStats:
    return BatchFeatureStats(
        model_id=model_id,
        window_start=datetime.utcnow() - timedelta(minutes=5),
        window_end=datetime.utcnow(),
        features=features,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def continuous_reference() -> FeatureDistributionStats:
    return make_continuous(loc=0.0, scale=1.0, seed=0, shap=0.15)


@pytest.fixture
def continuous_no_drift(continuous_reference) -> FeatureDistributionStats:
    rng = np.random.RandomState(1)
    data = rng.normal(0.01, 1.005, 1000)
    counts, _ = np.histogram(data, bins=continuous_reference.histogram_edges)
    return continuous_reference.model_copy(
        update={
            "window_start": datetime.utcnow() - timedelta(minutes=5),
            "window_end": datetime.utcnow(),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "histogram_counts": counts.tolist(),
            "shap_mean_abs": 0.16,
        }
    )


@pytest.fixture
def continuous_drifted() -> FeatureDistributionStats:
    drifted = make_continuous(loc=2.0, scale=1.8, seed=2, shap=0.45)
    return drifted.model_copy(
        update={
            "window_start": datetime.utcnow() - timedelta(minutes=5),
            "window_end": datetime.utcnow(),
        }
    )


@pytest.fixture
def categorical_reference() -> FeatureDistributionStats:
    return make_categorical()


@pytest.fixture
def categorical_drifted() -> FeatureDistributionStats:
    return make_categorical(
        value_counts={"north": 600, "south": 100, "east": 200, "west": 100}
    )
