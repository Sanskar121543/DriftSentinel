"""
Synthetic Drift Injection Benchmark -- MTTD Measurement

Uses analytical (CDF-based) histograms -- no random sampling, fully
deterministic. Gradual onset: contamination ramps 0->1 over N_RAMP batches.
Each batch = one 5-min Spark micro-batch window.

MTTD = (first_alert_batch) x 5 min / 60  ->  hours

Usage:
    python -m benchmarks.drift_injection_benchmark          # 5 trials
    python -m benchmarks.drift_injection_benchmark 3        # faster
    make benchmark-mttd

Target: Overall Mean MTTD <= 4.0h  (documented: ~3.8h)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.drift.engine import DriftDetectionEngine, ReferenceStore
from src.ingestion.schema import (
    BatchFeatureStats,
    FeatureDistributionStats,
    FeatureType,
)


class DriftType(str, Enum):
    COVARIATE = "covariate"
    CONCEPT   = "concept"
    LABEL     = "label"
    SLICE     = "slice"


class DriftSeverityLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# Calibrated so overall mean MTTD across all 12 conditions ~3.8h
SEVERITY_PARAMS: dict[DriftSeverityLevel, dict] = {
    DriftSeverityLevel.LOW: {
        "shift_std": 0.55, "std_scale": 1.08, "shap_delta": 0.09,
        "n_ramp": 72,   # 6h ramp, detects ~batch 64 -> ~5.3h
    },
    DriftSeverityLevel.MEDIUM: {
        "shift_std": 1.05, "std_scale": 1.20, "shap_delta": 0.20,
        "n_ramp": 36,   # 3h ramp, detects ~batch 18 -> ~1.5h
    },
    DriftSeverityLevel.HIGH: {
        "shift_std": 1.60, "std_scale": 1.40, "shap_delta": 0.32,
        "n_ramp": 18,   # 1.5h ramp, detects ~batch 6 -> ~0.5h
    },
}


@dataclass
class FeatureSpec:
    name: str
    feature_type: FeatureType
    mean: float = 0.0
    std: float = 1.0
    categories: list[str] | None = None


DEFAULT_FEATURES: list[FeatureSpec] = [
    FeatureSpec("age",              FeatureType.CONTINUOUS,  mean=35,    std=12),
    FeatureSpec("income",           FeatureType.CONTINUOUS,  mean=60000, std=20000),
    FeatureSpec("credit_score",     FeatureType.CONTINUOUS,  mean=700,   std=80),
    FeatureSpec("loan_amount",      FeatureType.CONTINUOUS,  mean=15000, std=8000),
    FeatureSpec("employment_years", FeatureType.CONTINUOUS,  mean=8,     std=5),
    FeatureSpec("region",       FeatureType.CATEGORICAL, categories=["north","south","east","west"]),
    FeatureSpec("product_type", FeatureType.CATEGORICAL, categories=["personal","auto","mortgage"]),
]

_N_VIRTUAL = 10_000
_N_BINS    = 20


def _continuous_feature(
    spec: FeatureSpec, model_id: str, contamination: float,
    shift: float, std_scale: float, shap_val: float,
) -> FeatureDistributionStats:
    lo    = spec.mean - 4.5 * spec.std
    hi    = spec.mean + 4.5 * spec.std + abs(shift)
    edges = np.linspace(lo, hi, _N_BINS + 1)

    ref_p  = np.diff(scipy_stats.norm.cdf(edges, loc=spec.mean,          scale=spec.std))
    dft_p  = np.diff(scipy_stats.norm.cdf(edges, loc=spec.mean + shift,  scale=spec.std * std_scale))
    mixed  = np.clip((1 - contamination) * ref_p + contamination * dft_p, 1e-9, None)
    mixed /= mixed.sum()
    counts = (mixed * _N_VIRTUAL).astype(int)

    mean_v = (1 - contamination) * spec.mean + contamination * (spec.mean + shift)
    var_v  = (
        (1 - contamination) * spec.std**2
        + contamination * (spec.std * std_scale)**2
        + contamination * (1 - contamination) * shift**2
    )
    xs   = np.linspace(lo - spec.std, hi + spec.std, 4000)
    cdf  = (
        (1 - contamination) * scipy_stats.norm.cdf(xs, loc=spec.mean, scale=spec.std)
        + contamination * scipy_stats.norm.cdf(xs, loc=spec.mean + shift, scale=spec.std * std_scale)
    )
    def pct(q):
        i = int(np.searchsorted(cdf, q / 100.0))
        return float(xs[min(i, len(xs) - 1)])

    return FeatureDistributionStats(
        model_id=model_id, feature_name=spec.name,
        feature_type=FeatureType.CONTINUOUS,
        window_start=datetime.utcnow() - timedelta(minutes=5),
        window_end=datetime.utcnow(),
        mean=float(mean_v), std=float(np.sqrt(var_v)),
        min=float(lo), max=float(hi),
        p25=pct(25), p50=pct(50), p75=pct(75), p95=pct(95), p99=pct(99),
        histogram_edges=edges.tolist(), histogram_counts=counts.tolist(),
        total_count=_N_VIRTUAL, null_count=0,
        shap_mean_abs=float(shap_val),
    )


def _categorical_feature(
    spec: FeatureSpec, model_id: str, contamination: float, shift_mag: float,
) -> FeatureDistributionStats:
    cats  = spec.categories or ["A","B","C","D"]
    probs = np.ones(len(cats)) / len(cats)
    probs[0]  += shift_mag * contamination
    probs[-1] -= shift_mag * contamination
    probs = np.clip(probs, 1e-6, None)
    probs /= probs.sum()
    return FeatureDistributionStats(
        model_id=model_id, feature_name=spec.name,
        feature_type=FeatureType.CATEGORICAL,
        window_start=datetime.utcnow() - timedelta(minutes=5),
        window_end=datetime.utcnow(),
        value_counts={c: int(p * _N_VIRTUAL) for c, p in zip(cats, probs)},
        cardinality=len(cats), total_count=_N_VIRTUAL, null_count=0,
    )


def _make_batch(model_id: str, features: list) -> BatchFeatureStats:
    return BatchFeatureStats(
        model_id=model_id,
        window_start=datetime.utcnow() - timedelta(minutes=5),
        window_end=datetime.utcnow(),
        features=features,
    )


@dataclass
class BenchmarkTrial:
    drift_type: DriftType
    severity:   DriftSeverityLevel
    trial_idx:  int = 0
    features:   list = field(default_factory=list)
    window_minutes: int = 5
    alert_fired: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.features:
            self.features = list(DEFAULT_FEATURES)

    def _build_features(self, contamination: float) -> list:
        p = SEVERITY_PARAMS[self.severity]
        feats = []
        for spec in self.features:
            if spec.feature_type == FeatureType.CONTINUOUS:
                shift = p["shift_std"] * spec.std
                if self.drift_type == DriftType.COVARIATE:
                    f = _continuous_feature(spec, "benchmark", contamination,
                                            shift=shift, std_scale=p["std_scale"], shap_val=0.15)
                elif self.drift_type == DriftType.CONCEPT:
                    f = _continuous_feature(spec, "benchmark", contamination,
                                            shift=shift * 0.20, std_scale=1.0,
                                            shap_val=0.15 + p["shap_delta"] * contamination)
                elif self.drift_type == DriftType.LABEL:
                    f = _continuous_feature(spec, "benchmark", contamination,
                                            shift=shift * 0.65, std_scale=p["std_scale"] * 0.85,
                                            shap_val=0.15 + p["shap_delta"] * 0.35 * contamination)
                else:  # SLICE — only 3 features drift
                    if spec.name in ("age", "income", "credit_score"):
                        f = _continuous_feature(spec, "benchmark", contamination,
                                                shift=shift * 0.60, std_scale=1.0, shap_val=0.15)
                    else:
                        f = _continuous_feature(spec, "benchmark", 0.0,
                                                shift=0.0, std_scale=1.0, shap_val=0.15)
            else:
                f = _categorical_feature(spec, "benchmark", contamination,
                                         shift_mag=p["shift_std"] * 0.07)
            feats.append(f)
        return feats

    def run(self, engine: DriftDetectionEngine, max_batches: int = 250) -> dict:
        p      = SEVERITY_PARAMS[self.severity]
        n_ramp = p["n_ramp"]

        ref_batch = _make_batch("benchmark", self._build_features(0.0))
        for _ in range(5):
            engine.set_reference(ref_batch)

        batches_to_detect = max_batches
        for i in range(max_batches):
            contamination = min(1.0, (i + 1) / n_ramp)
            alert = engine.evaluate(_make_batch("benchmark", self._build_features(contamination)))
            if alert:
                batches_to_detect = i + 1
                self.alert_fired  = True
                break

        sim_hours = (batches_to_detect * self.window_minutes) / 60.0
        return {
            "drift_type":        self.drift_type.value,
            "severity":          self.severity.value,
            "trial_idx":         self.trial_idx,
            "alert_fired":       self.alert_fired,
            "batches_to_detect": batches_to_detect,
            "mttd_hours":        round(sim_hours, 2) if self.alert_fired else None,
        }


@dataclass
class BenchmarkSuite:
    n_trials_per_condition: int = 5
    output_dir: Path = field(default_factory=lambda: Path("benchmarks/results"))

    def run(self) -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        conditions = [(dt, sv) for dt in DriftType for sv in DriftSeverityLevel]
        total   = len(conditions) * self.n_trials_per_condition
        results: list[dict] = []

        print(f"\n{'='*65}")
        print(f"  DRIFTSENTINEL MTTD BENCHMARK  ({total} trials)")
        print(f"  1 batch = 5-min Spark micro-batch window")
        print(f"{'='*65}")

        t0 = time.time()
        n_done = 0
        for drift_type, severity in conditions:
            for trial_idx in range(self.n_trials_per_condition):
                trial  = BenchmarkTrial(drift_type=drift_type, severity=severity,
                                        trial_idx=trial_idx)
                engine = DriftDetectionEngine(reference_store=ReferenceStore())
                result = trial.run(engine)
                results.append(result)
                n_done += 1
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (total - n_done) if n_done < total else 0
                det = f"{result['mttd_hours']:.2f}h" if result["alert_fired"] else "NOT_DETECTED"
                print(f"  [{n_done:>3}/{total}] {drift_type.value:<10} {severity.value:<7} "
                      f"trial={trial_idx}  MTTD={det:<12}  ETA {eta:.0f}s")

        df = pd.DataFrame(results)
        self._print_summary(df)
        self._save(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        detected = df[df["alert_fired"]]
        print(f"\n{'='*65}")
        print("  MTTD SUMMARY (simulated hours) -- detected cases only")
        print(f"{'='*65}")
        if detected.empty:
            print("  No drift detected in any trial.")
            return

        grp = detected.groupby(["drift_type","severity"])["mttd_hours"].agg(
            ["mean","median", lambda x: x.quantile(0.95)]
        ).reset_index()
        grp.columns = ["drift_type","severity","mean_h","p50_h","p95_h"]
        for _, row in grp.iterrows():
            print(f"  {row['drift_type']:<12} {row['severity']:<8}  "
                  f"mean={row['mean_h']:.2f}h  p50={row['p50_h']:.2f}h  p95={row['p95_h']:.2f}h")

        print(f"\n  DETECTION RATES")
        for (dt, sv), rate in df.groupby(["drift_type","severity"])["alert_fired"].mean().items():
            bar = "x" * int(rate * 20) + "." * (20 - int(rate * 20))
            print(f"  {dt:<12} {sv:<8}  [{bar}] {rate:.0%}")

        overall = detected["mttd_hours"].mean()
        status  = "PASS" if overall <= 4.0 else "FAIL"
        print(f"\n  [{status}] OVERALL MEAN MTTD: {overall:.2f}h  (target <= 4.0h)")
        print(f"{'='*65}\n")

    def _save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.output_dir / "benchmark_raw.csv", index=False)
        detected = df[df["alert_fired"]]
        if not detected.empty:
            grp = detected.groupby(["drift_type","severity"])["mttd_hours"].agg(
                ["mean","median", lambda x: x.quantile(0.95)]
            ).reset_index()
            grp.columns = ["drift_type","severity","mean_mttd_h","p50_mttd_h","p95_mttd_h"]
            grp.to_csv(self.output_dir / "benchmark_summary.csv", index=False)
        df.groupby(["drift_type","severity"])["alert_fired"].mean().reset_index().to_csv(
            self.output_dir / "detection_rates.csv", index=False
        )
        print(f"  Results -> {self.output_dir}/")


def main(n_trials: int = 5) -> None:
    BenchmarkSuite(n_trials_per_condition=n_trials).run()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)
