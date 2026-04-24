"""
DriftSentinel -- Standalone End-to-End Demo

Runs the full loop without Kafka, Spark, MLflow, or Docker.

    python demo.py
    make demo
"""
from __future__ import annotations
import uuid
from datetime import datetime, timedelta
import numpy as np

print("\n" + "="*65)
print("  DriftSentinel -- End-to-End Demo  (no external services)")
print("="*65)

# ── 1. Build reference and drifted batches ──────────────────────────────
print("\nSTEP 1: Building feature distributions ...")

from src.ingestion.schema import (
    BatchFeatureStats, FeatureDistributionStats, FeatureType, SLATier
)

rng = np.random.RandomState(42)

def make_cont(model_id, name, mean, std, rng, n=1000,
              shift=0.0, scale=1.0, shap=0.15):
    data  = rng.normal(mean + shift, std * scale, n)
    edges = np.linspace(mean - 4*std, mean + 4*std + abs(shift), 21).tolist()
    counts, _ = np.histogram(data, bins=edges)
    return FeatureDistributionStats(
        model_id=model_id, feature_name=name,
        feature_type=FeatureType.CONTINUOUS,
        window_start=datetime.utcnow() - timedelta(minutes=5),
        window_end=datetime.utcnow(),
        mean=float(data.mean()), std=float(data.std()),
        min=float(data.min()), max=float(data.max()),
        p25=float(np.percentile(data,25)), p50=float(np.percentile(data,50)),
        p75=float(np.percentile(data,75)), p95=float(np.percentile(data,95)),
        p99=float(np.percentile(data,99)),
        histogram_edges=edges, histogram_counts=counts.tolist(),
        total_count=n, null_count=0, shap_mean_abs=shap,
    )

def make_cat(model_id, name, cats, rng, n=1000, shift=None):
    k     = len(cats)
    probs = np.ones(k)/k
    if shift:
        for i, d in shift.items(): probs[i] += d
        probs = np.clip(probs, 0.01, None); probs /= probs.sum()
    samp  = rng.choice(cats, size=n, p=probs)
    vc    = {c: int((samp==c).sum()) for c in cats}
    return FeatureDistributionStats(
        model_id=model_id, feature_name=name,
        feature_type=FeatureType.CATEGORICAL,
        window_start=datetime.utcnow()-timedelta(minutes=5),
        window_end=datetime.utcnow(),
        value_counts=vc, cardinality=k, total_count=n, null_count=0,
    )

def batch(model_id, feats):
    return BatchFeatureStats(
        model_id=model_id,
        window_start=datetime.utcnow()-timedelta(minutes=10),
        window_end=datetime.utcnow()-timedelta(minutes=5),
        features=feats,
    )

ref_feats = [
    make_cont("credit_model","age",35,12,rng),
    make_cont("credit_model","income",60000,20000,rng),
    make_cont("credit_model","credit_score",700,80,rng),
    make_cont("credit_model","loan_amount",15000,8000,rng),
    make_cont("credit_model","employment_years",8,5,rng),
    make_cat("credit_model","region",["north","south","east","west"],rng),
    make_cat("credit_model","product_type",["personal","auto","mortgage"],rng),
]

drift_feats = [
    make_cont("credit_model","age",35,12,rng, shift=8.0,  scale=1.4, shap=0.27),
    make_cont("credit_model","income",60000,20000,rng, shift=15000, scale=1.3),
    make_cont("credit_model","credit_score",700,80,rng, shift=40,   scale=1.2, shap=0.40),
    make_cont("credit_model","loan_amount",15000,8000,rng),
    make_cont("credit_model","employment_years",8,5,rng),
    make_cat("credit_model","region",["north","south","east","west"],rng,
             shift={0:0.15, 1:-0.15}),
    make_cat("credit_model","product_type",["personal","auto","mortgage"],rng,
             shift={2:0.20, 0:-0.20}),
]

ref_batch   = batch("credit_model", ref_feats)
drift_batch = batch("credit_model", drift_feats)
print(f"  Reference: {len(ref_feats)} features | Drifted: {len(drift_feats)} features")

# ── 2. GE data quality gate ──────────────────────────────────────────────
print("\nSTEP 2: Great Expectations data quality gate ...")
from pathlib import Path
import pandas as pd
from src.ingestion.ge_validator import GEValidator

validator = GEValidator(suite_store_path=Path("/tmp/ds_demo/ge_suites"))
validator.auto_register_from_reference("credit_model", ref_feats)

sample_df = pd.DataFrame({
    "age":              rng.normal(35, 12, 120),
    "income":           rng.normal(60000, 20000, 120),
    "credit_score":     rng.normal(700, 80, 120),
    "loan_amount":      rng.normal(15000, 8000, 120),
    "employment_years": rng.normal(8, 5, 120).clip(0),
    "region":           rng.choice(["north","south","east","west"], 120),
    "product_type":     rng.choice(["personal","auto","mortgage"], 120),
})
sample_df.loc[:2, "age"] = -5  # inject bad rows
clean_df, failure = validator.validate_batch("credit_model", sample_df, "demo_batch")
n_fail = len(failure.failed_expectations) if failure else 0
print(f"  {len(clean_df)}/{len(sample_df)} rows clean | {n_fail} failed expectations")

# ── 3. Drift detection engine ────────────────────────────────────────────
print("\nSTEP 3: Running 5-test drift detection engine ...")
from src.drift.engine import DriftDetectionEngine, ReferenceStore

engine = DriftDetectionEngine(reference_store=ReferenceStore(), min_alert_tests=2)
engine.set_reference(ref_batch)
alert = engine.evaluate(drift_batch)

if alert:
    print(f"  DRIFT DETECTED  severity={alert.severity.value.upper()}")
    print(f"  Drifted features : {', '.join(alert.drifted_features)}")
    print(f"  Tests fired      : {alert.tests_fired}/{alert.tests_total}")
    for r in alert.test_results:
        if r.drifted:
            pval = f"p={r.p_value:.4f}" if r.p_value else ""
            print(f"    {r.test_name:<16} {r.feature_name:<20} stat={r.statistic:.4f} {pval}")
else:
    print("  No alert (try increasing drift magnitude)")

# ── 4. Strategy selector ─────────────────────────────────────────────────
print("\nSTEP 4: Cost-aware retraining strategy selector ...")
from src.retraining.strategy_selector import StrategySelector

selector = StrategySelector()
trigger  = selector.select(
    alert=alert,
    training_data_path="/data/credit_model/train.parquet",
    data_availability_ratio=1.2,
    days_since_last_retrain=30,
    sla_tier=SLATier.STANDARD,
    estimated_data_size_gb=8.0,
)
print(f"  Strategy : {trigger.strategy.value.upper()}")
print(f"  Cost est : ${trigger.estimated_cost_usd:.2f}")
if trigger.segment_filter:
    print(f"  Segment  : {trigger.segment_filter}")

# ── 5. Retraining pipeline ───────────────────────────────────────────────
print("\nSTEP 5: Retraining pipeline (Airflow DAG standalone) ...")
from airflow.dags.retrain_dag import run_pipeline

result = run_pipeline({
    "alert_id":              str(uuid.uuid4()),
    "model_id":                "credit_model",
    "strategy":                trigger.strategy.value,
    "estimated_cost_usd":      trigger.estimated_cost_usd,
    "sla_tier":                trigger.sla_tier.value,
    "training_data_path":      "/data/credit_model/train.parquet",
    "segment_filter":          trigger.segment_filter,
    "temporal_weight_lambda":  trigger.temporal_weight_lambda,
})

# ── 6. SPRT canary simulation ────────────────────────────────────────────
print("\nSTEP 6: SPRT sequential canary promotion ...")
from src.canary.sprt import SPRT, SPRTConfig

sprt = SPRT(SPRTConfig(alpha=0.05, beta=0.10, mde=0.02))
champ_auc = result["eval"]["champion_auc"]
chal_auc  = result["eval"]["challenger_auc"]
rng2 = np.random.RandomState(7)
n_samples = 0
decision  = None
for _ in range(200):
    champ = rng2.binomial(1, champ_auc, 50).astype(float).tolist()
    chal  = rng2.binomial(1, chal_auc,  50).astype(float).tolist()
    r     = sprt.update(champ, chal)
    n_samples += 50
    if r.decision.value != "hold":
        decision = r; break

print(f"  Champion AUC  : {champ_auc:.4f}")
print(f"  Challenger AUC: {chal_auc:.4f}  ({(chal_auc-champ_auc)/champ_auc:+.1%})")
if decision:
    print(f"  SPRT decision : {decision.decision.value.upper()} after {n_samples} samples")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  DEMO COMPLETE")
print("="*65)
print(f"  GE gate    : {len(clean_df)}/{len(sample_df)} rows clean")
if alert:
    print(f"  Drift      : {alert.severity.value.upper()} | {len(alert.drifted_features)} features | {alert.tests_fired} tests fired")
print(f"  Strategy   : {trigger.strategy.value}")
print(f"  Retrain    : {result['elapsed_seconds']:.1f}s  AUC={result['train']['val_auc']:.4f}")
if decision:
    print(f"  SPRT canary: {decision.decision.value.upper()} after {n_samples} samples")
print("="*65 + "\n")
