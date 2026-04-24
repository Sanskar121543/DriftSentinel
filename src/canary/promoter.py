"""
Canary Promotion Controller

Manages a 4-stage traffic ramp for challenger models:
  5% → 20% → 50% → 100%

At each stage:
  1. Collect metrics for SPRT_MIN_SAMPLES observations
  2. Run SPRT on the primary metric (conversion rate, quality score, etc.)
  3. Also hard-check: p99 latency, error rate, business metric
  4. PROMOTE: advance to next stage (or full promotion if at 100%)
  5. HOLD: collect more data, re-evaluate on next cycle
  6. ROLLBACK: revert to champion, file Jira ticket, emit CanaryDecisionEvent

Traffic routing is controlled via a feature flag key in the Kubernetes
ConfigMap (updated via k8s API).  Actual traffic split is enforced by
the Istio VirtualService weight annotation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any

import mlflow
from kubernetes import client as k8s_client, config as k8s_config

from src.canary.sprt import SPRT, SPRTConfig, SPRTResult
from src.ingestion.schema import (
    CanaryDecision,
    CanaryDecisionEvent,
    CanaryStageMetrics,
)
from src.utils.config import settings
from src.utils.jira import create_jira_ticket
from src.utils.logging import get_logger

logger = get_logger(__name__)

CANARY_STAGES = [0.05, 0.20, 0.50, 1.00]
SPRT_MIN_SAMPLES = 500   # minimum observations per stage before SPRT is evaluated
MAX_STAGE_DURATION_HOURS = 24


@dataclass
class MetricSnapshot:
    timestamp: float
    prediction_quality: float   # e.g., AUC, F1, accuracy on labeled slice
    p50_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    business_metric: float      # CTR, conversion rate, etc.
    business_metric_name: str


@dataclass
class StageBoundary:
    """Hard regression thresholds — SPRT is bypassed if any are violated."""
    max_p99_latency_ms: float = 500.0
    max_error_rate: float = 0.01
    min_business_metric_relative: float = -0.02   # -2% is the floor


@dataclass
class CanaryPromoter:
    """
    Orchestrates a full canary promotion lifecycle.
    Instantiate once per challenger deployment.
    """

    model_id: str
    challenger_version: str
    champion_version: str
    deployment_id: str
    primary_metric_fn: Callable[[], tuple[list[float], list[float]]] = field(repr=False)
    """
    Callable that returns (champion_outcomes, challenger_outcomes) for
    the primary SPRT metric since the last call.  Called by the polling loop.
    """
    metric_snapshot_fn: Callable[[], MetricSnapshot] = field(repr=False)
    """Callable that returns current operational metrics snapshot."""

    sprt_config: SPRTConfig = field(default_factory=SPRTConfig)
    stage_boundary: StageBoundary = field(default_factory=StageBoundary)

    # Internal state
    _current_stage_idx: int = field(default=0, init=False)
    _sprt: SPRT = field(init=False)
    _stage_history: list[CanaryStageMetrics] = field(default_factory=list, init=False)
    _started_at: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self._sprt = SPRT(self.sprt_config)

    # ------------------------------------------------------------------
    # Main promotion loop
    # ------------------------------------------------------------------

    async def run(self) -> CanaryDecisionEvent:
        """
        Blocking promotion loop.  Runs until PROMOTE to 100% or ROLLBACK.
        Returns the final CanaryDecisionEvent.
        """
        logger.info(
            "canary_started",
            model_id=self.model_id,
            challenger=self.challenger_version,
            champion=self.champion_version,
            deployment_id=self.deployment_id,
        )

        with mlflow.start_run(run_name=f"canary-{self.deployment_id[:8]}"):
            mlflow.set_tags({
                "deployment_id": self.deployment_id,
                "model_id": self.model_id,
                "challenger_version": self.challenger_version,
                "champion_version": self.champion_version,
            })

            while self._current_stage_idx < len(CANARY_STAGES):
                traffic_pct = CANARY_STAGES[self._current_stage_idx]
                logger.info(
                    "canary_stage_start",
                    stage=self._current_stage_idx + 1,
                    traffic_pct=f"{traffic_pct:.0%}",
                )

                # Set traffic weight
                await self._set_traffic(traffic_pct)
                self._sprt.reset()

                stage_decision = await self._run_stage(traffic_pct)

                if stage_decision.decision == CanaryDecision.ROLLBACK:
                    await self._set_traffic(0.0)
                    return await self._finalize(CanaryDecision.ROLLBACK, stage_decision.reason)

                if stage_decision.decision == CanaryDecision.PROMOTE:
                    if self._current_stage_idx == len(CANARY_STAGES) - 1:
                        # Full promotion at 100%
                        return await self._finalize(CanaryDecision.PROMOTE, stage_decision.reason)
                    self._current_stage_idx += 1

            # Should not reach here
            return await self._finalize(CanaryDecision.PROMOTE, "All stages passed.")

    # ------------------------------------------------------------------
    # Stage evaluation
    # ------------------------------------------------------------------

    async def _run_stage(self, traffic_pct: float) -> SPRTResult:
        stage_start = time.time()
        poll_interval_secs = 60  # evaluate every minute

        while True:
            elapsed_hours = (time.time() - stage_start) / 3600
            if elapsed_hours >= MAX_STAGE_DURATION_HOURS:
                logger.warning(
                    "stage_timeout_rolling_back",
                    traffic_pct=traffic_pct,
                    elapsed_hours=elapsed_hours,
                )
                return SPRTResult(
                    decision=CanaryDecision.ROLLBACK,
                    llr=self._sprt.current_llr,
                    n_samples=self._sprt.n_samples,
                    upper_boundary=self.sprt_config.upper_boundary,
                    lower_boundary=self.sprt_config.lower_boundary,
                    reason=f"Stage timeout after {elapsed_hours:.1f}h at {traffic_pct:.0%} traffic.",
                )

            await asyncio.sleep(poll_interval_secs)

            # Collect primary metric outcomes
            champ_outcomes, chal_outcomes = self.primary_metric_fn()

            if len(champ_outcomes) == 0:
                continue  # No new data yet

            sprt_result = self._sprt.update(champ_outcomes, chal_outcomes)

            # Hard regression check (bypass SPRT)
            snapshot = self.metric_snapshot_fn()
            hard_fail_reason = self._check_hard_boundaries(snapshot)

            if hard_fail_reason:
                stage_metrics = self._record_stage_metrics(
                    traffic_pct, snapshot, sprt_result, CanaryDecision.ROLLBACK
                )
                logger.warning(
                    "hard_boundary_violated_rolling_back",
                    reason=hard_fail_reason,
                    stage=traffic_pct,
                )
                return SPRTResult(
                    decision=CanaryDecision.ROLLBACK,
                    llr=sprt_result.llr,
                    n_samples=sprt_result.n_samples,
                    upper_boundary=sprt_result.upper_boundary,
                    lower_boundary=sprt_result.lower_boundary,
                    reason=f"Hard boundary violated: {hard_fail_reason}",
                )

            stage_metrics = self._record_stage_metrics(
                traffic_pct, snapshot, sprt_result, sprt_result.decision
            )

            mlflow.log_metrics({
                f"stage_{self._current_stage_idx + 1}_sprt_llr": sprt_result.llr,
                f"stage_{self._current_stage_idx + 1}_n_samples": sprt_result.n_samples,
                f"stage_{self._current_stage_idx + 1}_p99_ms": snapshot.p99_latency_ms,
                f"stage_{self._current_stage_idx + 1}_error_rate": snapshot.error_rate,
            }, step=sprt_result.n_samples)

            logger.info(
                "stage_sprt_update",
                decision=sprt_result.decision.value,
                llr=round(sprt_result.llr, 3),
                n=sprt_result.n_samples,
                stage=traffic_pct,
                reason=sprt_result.reason,
            )

            if sprt_result.decision in (CanaryDecision.PROMOTE, CanaryDecision.ROLLBACK):
                return sprt_result

    def _check_hard_boundaries(self, snap: MetricSnapshot) -> str | None:
        b = self.stage_boundary
        if snap.p99_latency_ms > b.max_p99_latency_ms:
            return f"p99 latency {snap.p99_latency_ms:.0f}ms > threshold {b.max_p99_latency_ms:.0f}ms"
        if snap.error_rate > b.max_error_rate:
            return f"error rate {snap.error_rate:.2%} > threshold {b.max_error_rate:.2%}"
        return None

    def _record_stage_metrics(
        self,
        traffic_pct: float,
        snapshot: MetricSnapshot,
        sprt: SPRTResult,
        decision: CanaryDecision,
    ) -> CanaryStageMetrics:
        m = CanaryStageMetrics(
            stage_traffic_pct=traffic_pct,
            sample_size=sprt.n_samples,
            prediction_quality=snapshot.prediction_quality,
            p50_latency_ms=snapshot.p50_latency_ms,
            p99_latency_ms=snapshot.p99_latency_ms,
            error_rate=snapshot.error_rate,
            business_metric=snapshot.business_metric,
            business_metric_name=snapshot.business_metric_name,
            sprt_llr=sprt.llr,
            sprt_decision=decision,
        )
        self._stage_history.append(m)
        return m

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    async def _finalize(
        self, decision: CanaryDecision, reason: str
    ) -> CanaryDecisionEvent:
        jira_id: str | None = None

        if decision == CanaryDecision.ROLLBACK:
            await self._swap_champion(to=self.champion_version)
            jira_id = await _file_rollback_ticket(
                model_id=self.model_id,
                challenger=self.challenger_version,
                champion=self.champion_version,
                deployment_id=self.deployment_id,
                reason=reason,
                stage_history=self._stage_history,
            )
            logger.warning(
                "canary_rollback",
                model_id=self.model_id,
                reason=reason,
                jira_id=jira_id,
            )
        else:
            await self._swap_champion(to=self.challenger_version)
            logger.info(
                "canary_promoted",
                model_id=self.model_id,
                version=self.challenger_version,
            )

        mlflow.log_params({
            "final_decision": decision.value,
            "stages_completed": self._current_stage_idx + 1,
        })

        event = CanaryDecisionEvent(
            deployment_id=self.deployment_id,
            model_id=self.model_id,
            challenger_version=self.challenger_version,
            champion_version=self.champion_version,
            final_decision=decision,
            stage_history=self._stage_history,
            rollback_reason=reason if decision == CanaryDecision.ROLLBACK else None,
            jira_ticket_id=jira_id,
        )
        return event

    async def _set_traffic(self, pct: float) -> None:
        """Update Istio VirtualService weights via Kubernetes API."""
        try:
            _update_istio_weights(
                model_id=self.model_id,
                champion_weight=int((1 - pct) * 100),
                challenger_weight=int(pct * 100),
                namespace=settings.k8s.namespace,
            )
            logger.info("traffic_weight_set", challenger_pct=pct)
        except Exception as exc:
            logger.error("traffic_set_failed", error=str(exc))

    async def _swap_champion(self, to: str) -> None:
        try:
            _patch_deployment_version(
                model_id=self.model_id,
                version=to,
                namespace=settings.k8s.namespace,
            )
            logger.info("champion_swapped", new_champion=to)
        except Exception as exc:
            logger.error("champion_swap_failed", error=str(exc))


# ---------------------------------------------------------------------------
# K8s helpers
# ---------------------------------------------------------------------------

def _update_istio_weights(
    model_id: str, champion_weight: int, challenger_weight: int, namespace: str
) -> None:
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()

    custom = k8s_client.CustomObjectsApi()
    patch = {
        "spec": {
            "http": [{
                "route": [
                    {"destination": {"host": f"{model_id}-champion"}, "weight": champion_weight},
                    {"destination": {"host": f"{model_id}-challenger"}, "weight": challenger_weight},
                ]
            }]
        }
    }
    custom.patch_namespaced_custom_object(
        group="networking.istio.io",
        version="v1alpha3",
        namespace=namespace,
        plural="virtualservices",
        name=f"{model_id}-vs",
        body=patch,
    )


def _patch_deployment_version(model_id: str, version: str, namespace: str) -> None:
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()

    apps_v1 = k8s_client.AppsV1Api()
    patch = {"spec": {"template": {"metadata": {"labels": {"model-version": version}}}}}
    apps_v1.patch_namespaced_deployment(
        name=f"{model_id}-deployment",
        namespace=namespace,
        body=patch,
    )


# ---------------------------------------------------------------------------
# Jira helper
# ---------------------------------------------------------------------------

async def _file_rollback_ticket(
    model_id: str,
    challenger: str,
    champion: str,
    deployment_id: str,
    reason: str,
    stage_history: list[CanaryStageMetrics],
) -> str | None:
    try:
        stage_summary = "\n".join([
            f"  Stage {s.stage_traffic_pct:.0%}: "
            f"n={s.sample_size}, p99={s.p99_latency_ms:.0f}ms, "
            f"err={s.error_rate:.2%}, LLR={s.sprt_llr:.3f}, decision={s.sprt_decision.value}"
            for s in stage_history
        ])
        description = (
            f"*Model:* {model_id}\n"
            f"*Challenger:* {challenger}\n"
            f"*Champion:* {champion}\n"
            f"*Deployment:* {deployment_id}\n\n"
            f"*Rollback reason:* {reason}\n\n"
            f"*Stage history:*\n{stage_summary}"
        )
        return await create_jira_ticket(
            project=settings.jira.project_key,
            summary=f"[DriftSentinel] Canary rollback: {model_id} v{challenger}",
            description=description,
            issue_type="Bug",
            labels=["ml-ops", "drift-sentinel", "canary-rollback"],
        )
    except Exception as exc:
        logger.error("jira_ticket_failed", error=str(exc))
        return None
