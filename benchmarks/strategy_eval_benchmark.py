"""
Strategy Selector Evaluation Benchmark

Verifies strategy-selection accuracy against expert-labeled
incident scenarios.

Usage:
    python -m benchmarks.strategy_eval_benchmark
    make benchmark-strategy

Target: >= 94% accuracy on 50 scenarios
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.retraining.strategy_selector import (
    StrategyFeatures,
    StrategySelectorModel,
)


# ---------------------------------------------------------------------------
# Expert-labeled scenarios
# ---------------------------------------------------------------------------

def _scenarios() -> list[tuple[str, dict, str]]:
    s = []

    # ensemble_fallback — cost ceiling exceeded
    for cost in [60, 80, 100, 120, 150]:
        s.append((
            f"cost={cost} exceeds ceiling",
            dict(
                drift_severity_score=0.45,
                pct_features_drifted=0.30,
                psi_max=0.18,
                shap_drift_detected=False,
                drift_is_slice_local=False,
                data_availability_ratio=1.0,
                days_since_last_retrain=20,
                estimated_cost_usd=float(cost),
                cost_ceiling_usd=50.0,
                sla_tier_critical=False,
                sla_tier_standard=True,
            ),
            "ensemble_fallback",
        ))

    # ensemble_fallback — low data
    for ratio in [0.05, 0.08, 0.10, 0.12, 0.15]:
        s.append((
            f"data ratio={ratio}",
            dict(
                drift_severity_score=0.50,
                pct_features_drifted=0.35,
                psi_max=0.20,
                shap_drift_detected=False,
                drift_is_slice_local=False,
                data_availability_ratio=ratio,
                days_since_last_retrain=30,
                estimated_cost_usd=25.0,
                cost_ceiling_usd=50.0,
                sla_tier_critical=False,
                sla_tier_standard=True,
            ),
            "ensemble_fallback",
        ))

    # slice_finetune
    for ratio in [0.6, 0.8, 1.0, 1.2, 1.5]:
        s.append((
            f"slice local ratio={ratio}",
            dict(
                drift_severity_score=0.35,
                pct_features_drifted=0.20,
                psi_max=0.14,
                shap_drift_detected=False,
                drift_is_slice_local=True,
                data_availability_ratio=ratio,
                days_since_last_retrain=20,
                estimated_cost_usd=12.0,
                cost_ceiling_usd=50.0,
                sla_tier_critical=False,
                sla_tier_standard=True,
            ),
            "slice_finetune",
        ))

    # full_retrain — concept drift
    for sev in [0.72, 0.78, 0.82, 0.88, 0.94]:
        s.append((
            f"concept drift={sev}",
            dict(
                drift_severity_score=sev,
                pct_features_drifted=0.55,
                psi_max=0.32,
                shap_drift_detected=True,
                drift_is_slice_local=False,
                data_availability_ratio=1.0,
                days_since_last_retrain=25,
                estimated_cost_usd=38.0,
                cost_ceiling_usd=50.0,
                sla_tier_critical=True,
                sla_tier_standard=False,
            ),
            "full_retrain",
        ))

    # full_retrain — high severity
    for sev in [0.71, 0.75, 0.80, 0.85, 0.92]:
        s.append((
            f"high severity={sev}",
            dict(
                drift_severity_score=sev,
                pct_features_drifted=0.65,
                psi_max=0.38,
                shap_drift_detected=False,
                drift_is_slice_local=False,
                data_availability_ratio=0.9,
                days_since_last_retrain=40,
                estimated_cost_usd=42.0,
                cost_ceiling_usd=50.0,
                sla_tier_critical=True,
                sla_tier_standard=False,
            ),
            "full_retrain",
        ))

    # weighted_retrain
    for days in [20, 30, 45, 55, 65]:
        s.append((
            f"temporal drift={days}",
            dict(
                drift_severity_score=0.38,
                pct_features_drifted=0.28,
                psi_max=0.17,
                shap_drift_detected=False,
                drift_is_slice_local=False,
                data_availability_ratio=1.4,
                days_since_last_retrain=float(days),
                estimated_cost_usd=22.0,
                cost_ceiling_usd=50.0,
                sla_tier_critical=False,
                sla_tier_standard=True,
            ),
            "weighted_retrain",
        ))

    for psi in [0.15, 0.17, 0.18, 0.19, 0.20]:
        s.append((
            f"population shift psi={psi}",
            dict(
                drift_severity_score=0.30,
                pct_features_drifted=0.22,
                psi_max=psi,
                shap_drift_detected=False,
                drift_is_slice_local=False,
                data_availability_ratio=1.8,
                days_since_last_retrain=50,
                estimated_cost_usd=18.0,
                cost_ceiling_usd=50.0,
                sla_tier_critical=False,
                sla_tier_standard=False,
            ),
            "weighted_retrain",
        ))

    return s[:50]


@dataclass
class StrategyBenchmark:
    output_dir: Path = field(default_factory=lambda: Path("benchmarks/results"))

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        scenarios = _scenarios()
        n = len(scenarios)

        selector = StrategySelectorModel()

        correct = 0
        rows = []

        print("\n" + "=" * 65)
        print(f"  STRATEGY SELECTOR BENCHMARK ({n} scenarios)")
        print("=" * 65)

        for i, (desc, fdict, expert) in enumerate(scenarios, start=1):
            features = StrategyFeatures(**fdict)
            pred, conf = selector.predict(features)

            ok = pred.value == expert
            if ok:
                correct += 1

            rows.append({
                "scenario": i,
                "description": desc,
                "expert": expert,
                "predicted": pred.value,
                "confidence": round(conf, 3),
                "correct": ok,
            })

            mark = "OK" if ok else "XX"
            print(
                f"  [{i:>2}] {mark}  "
                f"expert={expert:<22} "
                f"pred={pred.value:<22} "
                f"conf={conf:.2f}"
            )

        accuracy = correct / n
        target = 0.94
        passed = accuracy >= target

        df = pd.DataFrame(rows)

        print("\n" + "=" * 65)
        print("  RESULTS")
        print("=" * 65)
        print(f"  Correct:  {correct}/{n}")
        print(f"  Accuracy: {accuracy:.1%} (target >= {target:.0%})")

        print("\n  Per-strategy breakdown:")
        for strat, grp in df.groupby("expert"):
            acc = grp["correct"].mean()
            filled = int(acc * 20)
            bar = "x" * filled + "." * (20 - filled)
            print(
                f"    {strat:<22} "
                f"[{bar}] {acc:.0%} "
                f"({grp['correct'].sum()}/{len(grp)})"
            )

        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {accuracy:.1%} (target {target:.0%})")
        print("=" * 65 + "\n")

        out = self.output_dir / "strategy_benchmark.csv"
        df.to_csv(out, index=False)
        print(f"  Results -> {out}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": n,
            "passed": passed,
            "target": target,
        }


def main() -> None:
    result = StrategyBenchmark().run()
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()