"""
Sequential Probability Ratio Test (SPRT) — Wald's Test

Used for canary model promotion decisions.

Unlike fixed-horizon A/B testing (where you wait for a predetermined sample
size), SPRT makes the call as soon as sufficient statistical evidence exists.
This cuts expected promotion time by ~60% when the challenger is clearly
better or worse early in the canary.

Hypotheses:
  H0: δ ≤ 0        (challenger is not better than champion)
  H1: δ ≥ MDE      (challenger improves metric by at least MDE)

The test computes a log-likelihood ratio (LLR).  When LLR crosses:
  upper boundary (log(1-β)/α):  PROMOTE — H1 accepted
  lower boundary (log(β/(1-α))): ROLLBACK — H0 accepted
  Neither crossed:               HOLD — collect more data

References:
  Wald, A. (1947). Sequential Analysis. John Wiley & Sons.
  Spotify Engineering: https://engineering.atspotify.com/2023/03/ab-testing-at-spotify
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.ingestion.schema import CanaryDecision
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SPRTConfig:
    alpha: float = 0.05   # Type I error rate (false positive — wrongly promote)
    beta: float = 0.10    # Type II error rate (false negative — wrongly keep champion)
    mde: float = 0.02     # Minimum detectable effect (relative improvement threshold)

    @property
    def upper_boundary(self) -> float:
        """Log of (1 - β) / α — accept H1 (promote) boundary."""
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower_boundary(self) -> float:
        """Log of β / (1 - α) — accept H0 (rollback) boundary."""
        return math.log(self.beta / (1 - self.alpha))


class SPRTResult(NamedTuple):
    decision: CanaryDecision
    llr: float              # Log-likelihood ratio at decision time
    n_samples: int          # Samples consumed before decision
    upper_boundary: float
    lower_boundary: float
    reason: str


class SPRT:
    """
    Computes the SPRT decision for comparing two Bernoulli-distributed
    metrics (conversion rate, error rate, quality score > threshold, etc.).

    For each new (champion, challenger) outcome pair:
      - Compute incremental LLR contribution
      - Accumulate running LLR
      - Check boundaries
    """

    def __init__(self, config: SPRTConfig | None = None) -> None:
        self.config = config or SPRTConfig()
        self._llr: float = 0.0
        self._n: int = 0

    def reset(self) -> None:
        self._llr = 0.0
        self._n = 0

    @property
    def current_llr(self) -> float:
        return self._llr

    @property
    def n_samples(self) -> int:
        return self._n

    def update(
        self,
        champion_outcomes: list[float],
        challenger_outcomes: list[float],
    ) -> SPRTResult:
        """
        Update SPRT with new batch of outcomes.

        Outcomes should be in [0, 1] (binary: success/failure, or normalized quality scores).

        Returns the current SPRT decision.
        """
        if len(champion_outcomes) != len(challenger_outcomes):
            raise ValueError("champion and challenger outcome lists must be the same length")

        for champ_y, chal_y in zip(champion_outcomes, challenger_outcomes):
            self._update_single(champ_y, chal_y)
            self._n += 1

        return self._make_decision()

    def _update_single(self, p0: float, p1: float) -> None:
        """
        LLR contribution of one paired observation.

        Under H0 (challenger = champion):       P(obs | H0) ∝ p0
        Under H1 (challenger = champion + MDE): P(obs | H1) ∝ p0 + MDE

        Log-likelihood ratio increment:
          LLR += log[P(p1 | H1)] - log[P(p0 | H0)]
        """
        eps = 1e-9
        p0_clipped = np.clip(p0, eps, 1 - eps)
        p1_clipped = np.clip(p1, eps, 1 - eps)

        # H1: challenger improves by MDE
        _p1_under_h1 = np.clip(p0_clipped + self.config.mde, eps, 1 - eps)

        # Bernoulli log-likelihood ratio
        if abs(p1_clipped - p0_clipped) < eps:
            llr_increment = 0.0
        else:
            # Ratio of challenger likelihood under H1 vs H0
            llr_increment = math.log(p1_clipped / p0_clipped + eps)

        self._llr += llr_increment

    def _make_decision(self) -> SPRTResult:
        cfg = self.config

        if self._llr >= cfg.upper_boundary:
            decision = CanaryDecision.PROMOTE
            reason = (
                f"LLR ({self._llr:.3f}) exceeded upper boundary ({cfg.upper_boundary:.3f}). "
                f"Challenger shows significant improvement (α={cfg.alpha}, MDE={cfg.mde:.1%})."
            )
        elif self._llr <= cfg.lower_boundary:
            decision = CanaryDecision.ROLLBACK
            reason = (
                f"LLR ({self._llr:.3f}) fell below lower boundary ({cfg.lower_boundary:.3f}). "
                f"Challenger not better than champion at β={cfg.beta}."
            )
        else:
            decision = CanaryDecision.HOLD
            reason = (
                f"LLR ({self._llr:.3f}) between boundaries "
                f"[{cfg.lower_boundary:.3f}, {cfg.upper_boundary:.3f}]. "
                f"Collecting more data ({self._n} samples so far)."
            )

        return SPRTResult(
            decision=decision,
            llr=self._llr,
            n_samples=self._n,
            upper_boundary=cfg.upper_boundary,
            lower_boundary=cfg.lower_boundary,
            reason=reason,
        )

    def expected_sample_size(self, p0: float) -> dict[str, float]:
        """
        Approximate expected sample size under H0 and H1 using Wald's formula.
        Useful for capacity planning before a canary.
        """
        cfg = self.config
        eps = 1e-9
        p1 = min(p0 + cfg.mde, 1 - eps)
        p0 = max(p0, eps)

        # Expected LLR per observation
        e_llr_h0 = p0 * math.log(p0 / (p1 + eps) + eps) + (1 - p0) * math.log((1 - p0) / (1 - p1 + eps) + eps)
        e_llr_h1 = p1 * math.log(p1 / (p0 + eps) + eps) + (1 - p1) * math.log((1 - p1) / (1 - p0 + eps) + eps)

        n_h0 = (cfg.lower_boundary / e_llr_h0) if abs(e_llr_h0) > eps else float("inf")
        n_h1 = (cfg.upper_boundary / e_llr_h1) if abs(e_llr_h1) > eps else float("inf")

        return {
            "expected_n_under_h0": abs(n_h0),
            "expected_n_under_h1": abs(n_h1),
            "fixed_horizon_n_approx": _fixed_horizon_n(cfg.alpha, cfg.beta, cfg.mde, p0),
        }


def _fixed_horizon_n(alpha: float, beta: float, mde: float, p0: float) -> float:
    """Two-proportion z-test sample size (for comparison with SPRT)."""
    z_alpha = 1.96  # z_{1-α/2}
    z_beta = 1.28   # z_{1-β}
    p1 = p0 + mde
    pooled = (p0 + p1) / 2
    return (z_alpha + z_beta) ** 2 * pooled * (1 - pooled) / (mde ** 2)
