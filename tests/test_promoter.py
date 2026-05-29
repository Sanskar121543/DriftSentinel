"""
Canary promoter tests with mlflow + kubernetes stubbed out.

The promoter pulls in heavy infra clients (mlflow, kubernetes) at import time
and drives an async stage loop. These tests inject lightweight fakes into
sys.modules *before* import so the orchestration logic — hard-boundary
bypass, stage advancement, promote/rollback finalization — can be exercised
deterministically without any cluster or tracking server.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest


# ---------------------------------------------------------------------------
# Inject fake mlflow + kubernetes before importing the promoter
# ---------------------------------------------------------------------------

def _install_fakes():
    mlflow = types.ModuleType("mlflow")

    @contextmanager
    def _start_run(*_a, **_k):
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id="fake"))

    mlflow.start_run = _start_run
    mlflow.set_tags = lambda *_a, **_k: None
    mlflow.log_metrics = lambda *_a, **_k: None
    mlflow.log_params = lambda *_a, **_k: None
    sys.modules["mlflow"] = mlflow

    kubernetes = types.ModuleType("kubernetes")
    client = types.ModuleType("kubernetes.client")
    config = types.ModuleType("kubernetes.config")
    client.CustomObjectsApi = lambda *_a, **_k: types.SimpleNamespace(
        patch_namespaced_custom_object=lambda **_kw: None
    )
    client.AppsV1Api = lambda *_a, **_k: types.SimpleNamespace(
        patch_namespaced_deployment=lambda **_kw: None
    )
    config.load_incluster_config = lambda: None
    config.load_kube_config = lambda: None
    kubernetes.client = client
    kubernetes.config = config
    sys.modules["kubernetes"] = kubernetes
    sys.modules["kubernetes.client"] = client
    sys.modules["kubernetes.config"] = config


_install_fakes()

from src.canary.promoter import (  # noqa: E402
    CanaryPromoter,
    MetricSnapshot,
    StageBoundary,
)
from src.canary.sprt import SPRTConfig  # noqa: E402
from src.ingestion.schema import CanaryDecision  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(p99=100.0, err=0.001) -> MetricSnapshot:
    return MetricSnapshot(
        timestamp=0.0,
        prediction_quality=0.9,
        p50_latency_ms=20.0,
        p99_latency_ms=p99,
        error_rate=err,
        business_metric=0.3,
        business_metric_name="conversion",
    )


def _make_promoter(primary_fn, snapshot_fn) -> CanaryPromoter:
    return CanaryPromoter(
        model_id="m",
        challenger_version="v2",
        champion_version="v1",
        deployment_id="deadbeefcafebabe",
        primary_metric_fn=primary_fn,
        metric_snapshot_fn=snapshot_fn,
        sprt_config=SPRTConfig(alpha=0.05, beta=0.10, mde=0.02),
    )


# ---------------------------------------------------------------------------
# Hard boundary logic
# ---------------------------------------------------------------------------

class TestHardBoundaries:
    def test_latency_violation_detected(self):
        p = _make_promoter(lambda: ([], []), lambda: _snapshot())
        reason = p._check_hard_boundaries(_snapshot(p99=999))
        assert reason is not None and "latency" in reason

    def test_error_rate_violation_detected(self):
        p = _make_promoter(lambda: ([], []), lambda: _snapshot())
        reason = p._check_hard_boundaries(_snapshot(err=0.5))
        assert reason is not None and "error rate" in reason

    def test_healthy_snapshot_passes(self):
        p = _make_promoter(lambda: ([], []), lambda: _snapshot())
        assert p._check_hard_boundaries(_snapshot()) is None

    def test_custom_boundary_threshold(self):
        p = _make_promoter(lambda: ([], []), lambda: _snapshot())
        p.stage_boundary = StageBoundary(max_p99_latency_ms=50.0)
        assert p._check_hard_boundaries(_snapshot(p99=80)) is not None


# ---------------------------------------------------------------------------
# Stage metric recording
# ---------------------------------------------------------------------------

def test_record_stage_metrics_appends_history():
    p = _make_promoter(lambda: ([], []), lambda: _snapshot())
    from src.canary.sprt import SPRTResult

    sprt = SPRTResult(
        decision=CanaryDecision.HOLD,
        llr=0.5,
        n_samples=100,
        upper_boundary=2.9,
        lower_boundary=-2.3,
        reason="hold",
    )
    p._record_stage_metrics(0.05, _snapshot(), sprt, CanaryDecision.HOLD)
    assert len(p._stage_history) == 1
    assert p._stage_history[0].stage_traffic_pct == 0.05


# ---------------------------------------------------------------------------
# Full async lifecycle (sleep patched to no-op)
# ---------------------------------------------------------------------------

@pytest.fixture
def no_sleep(monkeypatch):
    async def _instant(_secs):
        return None

    import src.canary.promoter as promoter_mod

    monkeypatch.setattr(promoter_mod.asyncio, "sleep", _instant)
    return promoter_mod


@pytest.mark.asyncio
async def test_full_promotion_when_challenger_better(no_sleep):
    # Challenger consistently better → every stage promotes → final PROMOTE.
    p = _make_promoter(lambda: ([0.20] * 50, [0.30] * 50), lambda: _snapshot())
    event = await p.run()
    assert event.final_decision == CanaryDecision.PROMOTE
    assert event.rollback_reason is None


@pytest.mark.asyncio
async def test_rollback_on_hard_boundary(no_sleep):
    # First poll returns data but latency is over the hard ceiling → ROLLBACK.
    p = _make_promoter(lambda: ([0.20] * 50, [0.30] * 50), lambda: _snapshot(p99=999))
    event = await p.run()
    assert event.final_decision == CanaryDecision.ROLLBACK
    assert "Hard boundary" in event.rollback_reason


@pytest.mark.asyncio
async def test_rollback_when_challenger_worse(no_sleep):
    p = _make_promoter(lambda: ([0.30] * 50, [0.15] * 50), lambda: _snapshot())
    event = await p.run()
    assert event.final_decision == CanaryDecision.ROLLBACK
