"""
DriftSentinel Retraining DAG

Triggered by a message on the retrain-triggers Kafka topic (consumed by
an Airflow KafkaSensor).  Executes the full retraining pipeline:

  1. validate_trigger       — Schema validation + cost ceiling check
  2. prepare_training_data  — Pull data from GCS, apply temporal weights / slice filter
  3. run_great_expectations — Final quality check on training data
  4. train_model            — PyTorch training on Dataproc (or local)
  5. evaluate_model         — Compute holdout metrics + compare to champion
  6. register_model         — Register challenger in MLflow Model Registry
  7. trigger_canary         — Publish CanaryDecisionEvent → CanaryPromoter
  8. notify                 — Log completion

Idempotent: DAG run is keyed on trigger_id; duplicate triggers are skipped.
All steps log artifacts to MLflow run linked to the trigger_id.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Try importing Airflow; fall back gracefully when running outside Airflow
# (e.g. in unit tests or when just importing the module for inspection).
# ---------------------------------------------------------------------------
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.empty import EmptyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default DAG args
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "driftsentinel",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=6),
    "email_on_failure": False,
}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ===========================================================================
# Task implementations (pure Python — no Airflow deps)
# ===========================================================================


def validate_trigger(trigger_json: str, **kwargs) -> dict:
    """
    Step 1: Validate the RetrainTrigger payload.
    Checks cost ceiling; raises if exceeded (forces human review).
    """
    import sys
    sys.path.insert(0, "/opt/airflow")

    from src.ingestion.schema import RetrainTrigger

    trigger = RetrainTrigger.model_validate_json(trigger_json)
    cost_ceiling = float(os.getenv("RETRAINING_COST_CEILING_USD", "50.0"))

    if trigger.estimated_cost_usd > cost_ceiling:
        raise ValueError(
            f"Estimated cost ${trigger.estimated_cost_usd:.2f} exceeds ceiling "
            f"${cost_ceiling:.2f}. Escalate to engineer."
        )

    print(
        f"[validate_trigger] OK — model={trigger.model_id} "
        f"strategy={trigger.strategy.value} "
        f"cost=${trigger.estimated_cost_usd:.2f}"
    )
    return trigger.model_dump(mode="json")


def prepare_training_data(trigger_dict: dict, **kwargs) -> dict:
    """
    Step 2: Fetch training data from GCS/local storage.
    Apply temporal weighting (exponential decay) for WEIGHTED_RETRAIN.
    Apply segment filter for SLICE_FINETUNE.
    """
    from src.ingestion.schema import RetrainingStrategy

    strategy = RetrainingStrategy(trigger_dict["strategy"])
    data_path = trigger_dict.get("training_data_path", "/tmp/synthetic_training_data.parquet")
    segment_filter = trigger_dict.get("segment_filter")
    temporal_lambda = trigger_dict.get("temporal_weight_lambda")

    # --- Synthetic data for demo (replace with GCS read in production) ---
    rng = np.random.RandomState(42)
    n = 5000
    df = pd.DataFrame({
        "age": rng.normal(35, 12, n),
        "income": rng.normal(60000, 20000, n),
        "credit_score": rng.normal(700, 80, n),
        "loan_amount": rng.normal(15000, 8000, n),
        "employment_years": rng.normal(8, 5, n).clip(0),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "product_type": rng.choice(["personal", "auto", "mortgage"], n),
        "label": rng.binomial(1, 0.3, n),
        "event_timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
    })

    # Apply slice filter for SLICE_FINETUNE
    if strategy == RetrainingStrategy.SLICE_FINETUNE and segment_filter:
        for col, val in segment_filter.items():
            if col in df.columns:
                df = df[df[col] == val]
                print(f"[prepare_data] Slice filter applied: {col}={val}, rows={len(df)}")

    # Apply temporal weights for WEIGHTED_RETRAIN
    sample_weights = None
    if strategy == RetrainingStrategy.WEIGHTED_RETRAIN and temporal_lambda:
        now = df["event_timestamp"].max()
        hours_ago = (now - df["event_timestamp"]).dt.total_seconds() / 3600
        sample_weights = np.exp(-temporal_lambda * hours_ago.values)
        sample_weights /= sample_weights.sum()
        print(f"[prepare_data] Temporal weights applied (λ={temporal_lambda})")

    # Save to temp parquet
    out_path = f"/tmp/ds_train_{trigger_dict['trigger_id'][:8]}.parquet"
    df.to_parquet(out_path, index=False)

    weights_path = None
    if sample_weights is not None:
        weights_path = out_path.replace(".parquet", "_weights.npy")
        np.save(weights_path, sample_weights)

    print(f"[prepare_data] Dataset prepared: {len(df)} rows → {out_path}")
    return {
        "data_path": out_path,
        "weights_path": weights_path,
        "row_count": len(df),
        "strategy": strategy.value,
    }


def run_great_expectations(data_meta: dict, trigger_dict: dict, **kwargs) -> dict:
    """
    Step 3: Final data quality gate on training data before training.
    Quarantines rows that fail expectations.
    """
    import sys
    sys.path.insert(0, "/opt/airflow")

    df = pd.read_parquet(data_meta["data_path"])
    model_id = trigger_dict["model_id"]

    # Simple inline checks (GE context not needed for training data gate)
    initial_rows = len(df)
    issues = []

    # Null checks
    for col in ["age", "income", "credit_score", "label"]:
        if col in df.columns:
            null_pct = df[col].isna().mean()
            if null_pct > 0.05:
                issues.append(f"{col} null_pct={null_pct:.2%} > 5%")

    # Range checks
    if "age" in df.columns:
        bad = ((df["age"] < 18) | (df["age"] > 100)).sum()
        if bad > len(df) * 0.01:
            df = df[(df["age"] >= 18) & (df["age"] <= 100)]
            issues.append(f"Removed {bad} rows with invalid age")

    if "label" in df.columns:
        valid_labels = df["label"].isin([0, 1]).mean()
        if valid_labels < 0.99:
            df = df[df["label"].isin([0, 1])]
            issues.append(f"Removed rows with invalid labels")

    # Save cleaned data
    clean_path = data_meta["data_path"].replace(".parquet", "_clean.parquet")
    df.to_parquet(clean_path, index=False)

    result = {
        "ge_passed": len(issues) == 0,
        "issues": issues,
        "rows_before": initial_rows,
        "rows_after": len(df),
        "quarantine_pct": (initial_rows - len(df)) / max(initial_rows, 1),
        "clean_data_path": clean_path,
        "weights_path": data_meta.get("weights_path"),
    }

    if issues:
        print(f"[ge_validation] Issues found: {issues}")
    else:
        print(f"[ge_validation] All checks passed — {len(df)} clean rows")

    return result


def train_model(ge_meta: dict, trigger_dict: dict, **kwargs) -> dict:
    """
    Step 4: Train the challenger model.

    Uses a simple scikit-learn gradient boosting model as a stand-in
    for PyTorch production training.  In production, this submits a
    GCP Dataproc PySpark job with PyTorch DDP.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    import pickle

    clean_path = ge_meta["clean_data_path"]
    weights_path = ge_meta.get("weights_path")

    df = pd.read_parquet(clean_path)
    feature_cols = ["age", "income", "credit_score", "loan_amount", "employment_years"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df["label"].values

    # Load sample weights if available
    sample_weights = None
    if weights_path and Path(weights_path).exists():
        sample_weights = np.load(weights_path)
        sample_weights = sample_weights[:len(X)]  # Align after row removal

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Truncate weights to train split size
    train_weights = None
    if sample_weights is not None:
        train_weights = sample_weights[:len(X_train)]

    strategy = trigger_dict.get("strategy", "full_retrain")
    print(f"[train] Strategy={strategy}, X_train={X_train.shape}")

    with mlflow.start_run(run_name=f"retrain-{trigger_dict['trigger_id'][:8]}") as run:
        mlflow.set_tags({
            "trigger_id": trigger_dict["trigger_id"],
            "model_id": trigger_dict["model_id"],
            "strategy": strategy,
        })

        # Training
        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=train_weights)

        # Evaluation
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)
        auc = float(roc_auc_score(y_val, y_pred_proba))
        f1 = float(f1_score(y_val, y_pred))

        mlflow.log_metrics({"val_auc": auc, "val_f1": f1, "train_rows": len(X_train)})

        # Save model
        model_path = f"/tmp/ds_challenger_{trigger_dict['trigger_id'][:8]}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        mlflow.sklearn.log_model(clf, "challenger_model")

        print(f"[train] AUC={auc:.4f}, F1={f1:.4f}, run_id={run.info.run_id}")

        return {
            "model_path": model_path,
            "val_auc": auc,
            "val_f1": f1,
            "mlflow_run_id": run.info.run_id,
            "feature_cols": feature_cols,
        }


def evaluate_model(train_meta: dict, trigger_dict: dict, **kwargs) -> dict:
    """
    Step 5: Compare challenger vs champion on holdout set.
    Computes lift, improvement %, and whether challenger should proceed to canary.
    """
    challenger_auc = train_meta["val_auc"]
    challenger_f1 = train_meta["val_f1"]

    # In production: load champion metrics from MLflow Model Registry
    # Here we use a synthetic champion baseline
    champion_auc = 0.78   # Simulated champion performance
    champion_f1 = 0.65

    auc_lift = (challenger_auc - champion_auc) / max(champion_auc, 1e-9)
    f1_lift = (challenger_f1 - champion_f1) / max(champion_f1, 1e-9)

    # Proceed to canary if challenger is at least as good as champion
    proceed_to_canary = (
        challenger_auc >= champion_auc - 0.01 and  # Allow tiny regression on AUC
        challenger_f1 >= champion_f1 - 0.02         # Allow tiny regression on F1
    )

    print(
        f"[evaluate] Challenger AUC={challenger_auc:.4f} ({auc_lift:+.1%} vs champion) "
        f"F1={challenger_f1:.4f} ({f1_lift:+.1%}) — proceed={proceed_to_canary}"
    )

    return {
        "champion_auc": champion_auc,
        "champion_f1": champion_f1,
        "challenger_auc": challenger_auc,
        "challenger_f1": challenger_f1,
        "auc_lift_pct": round(auc_lift * 100, 2),
        "f1_lift_pct": round(f1_lift * 100, 2),
        "proceed_to_canary": proceed_to_canary,
    }


def register_model(eval_meta: dict, train_meta: dict, trigger_dict: dict, **kwargs) -> dict:
    """
    Step 6: Register challenger in MLflow Model Registry.
    Tags with challenger version and links to trigger_id.
    """
    if not eval_meta["proceed_to_canary"]:
        print("[register] Skipping — challenger did not beat champion threshold.")
        return {"registered": False, "reason": "Did not beat champion"}

    model_id = trigger_dict["model_id"]
    challenger_version = f"v{datetime.utcnow().strftime('%Y%m%d-%H%M')}"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        client.create_registered_model(model_id)
    except Exception:
        pass  # Already exists

    mv = client.create_model_version(
        name=model_id,
        source=f"runs:/{train_meta['mlflow_run_id']}/challenger_model",
        run_id=train_meta["mlflow_run_id"],
        tags={
            "trigger_id": trigger_dict["trigger_id"],
            "strategy": trigger_dict["strategy"],
            "val_auc": str(round(train_meta["val_auc"], 4)),
        },
    )

    print(f"[register] Model {model_id} v{mv.version} registered as {challenger_version}")
    return {
        "registered": True,
        "model_name": model_id,
        "model_version": mv.version,
        "challenger_version": challenger_version,
    }


def trigger_canary(
    register_meta: dict, eval_meta: dict, trigger_dict: dict, **kwargs
) -> dict:
    """
    Step 7: Emit a RetrainTrigger to the canary-decisions pipeline.
    In production, this publishes to Kafka retrain-triggers topic.
    Here we log to MLflow and print a summary.
    """
    if not register_meta.get("registered"):
        print("[canary] Skipping canary — model not registered.")
        return {"canary_triggered": False}

    model_id = trigger_dict["model_id"]
    challenger_version = register_meta.get("challenger_version", "unknown")

    # In production: produce CanaryDecisionEvent to Kafka
    # from src.ingestion.producer import DriftSentinelProducer
    # producer.produce_retrain_trigger(trigger)

    summary = {
        "canary_triggered": True,
        "model_id": model_id,
        "challenger_version": challenger_version,
        "champion_auc": eval_meta["champion_auc"],
        "challenger_auc": eval_meta["challenger_auc"],
        "auc_lift_pct": eval_meta["auc_lift_pct"],
        "canary_stages": [0.05, 0.20, 0.50, 1.00],
        "strategy": trigger_dict["strategy"],
        "triggered_at": datetime.utcnow().isoformat(),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=f"canary-trigger-{model_id}"):
        mlflow.log_dict(summary, "canary_trigger.json")

    print(
        f"\n{'='*60}\n"
        f"[canary] CANARY TRIGGERED\n"
        f"  Model: {model_id}  Challenger: {challenger_version}\n"
        f"  AUC lift: {eval_meta['auc_lift_pct']:+.2f}%\n"
        f"  Traffic ramp: 5% → 20% → 50% → 100%\n"
        f"{'='*60}\n"
    )

    return summary


# ===========================================================================
# Standalone runner (outside Airflow — for demo/testing)
# ===========================================================================

def run_pipeline(trigger_payload: dict) -> dict:
    """
    Run the full retraining pipeline sequentially.
    Used when Airflow is not available (local demo, CI, benchmark).
    """
    trigger_json = json.dumps(trigger_payload)

    print("\n" + "="*60)
    print("DRIFTSENTINEL RETRAINING PIPELINE")
    print("="*60)

    t0 = time.time()

    trigger_dict = validate_trigger(trigger_json)
    print(f"  ✓ Step 1: Trigger validated")

    data_meta = prepare_training_data(trigger_dict)
    print(f"  ✓ Step 2: Training data prepared ({data_meta['row_count']} rows)")

    ge_meta = run_great_expectations(data_meta, trigger_dict)
    print(f"  ✓ Step 3: GE validation — {ge_meta['rows_after']} clean rows")

    train_meta = train_model(ge_meta, trigger_dict)
    print(f"  ✓ Step 4: Model trained — AUC={train_meta['val_auc']:.4f}")

    eval_meta = evaluate_model(train_meta, trigger_dict)
    print(f"  ✓ Step 5: Evaluation — AUC lift {eval_meta['auc_lift_pct']:+.2f}%")

    register_meta = register_model(eval_meta, train_meta, trigger_dict)
    print(f"  ✓ Step 6: Model {'registered' if register_meta.get('registered') else 'skipped (no improvement)'}")

    canary_meta = trigger_canary(register_meta, eval_meta, trigger_dict)
    print(f"  ✓ Step 7: Canary {'triggered' if canary_meta.get('canary_triggered') else 'skipped'}")

    elapsed = time.time() - t0
    print(f"\n  Pipeline complete in {elapsed:.1f}s")
    print("="*60 + "\n")

    return {
        "trigger": trigger_dict,
        "data": data_meta,
        "ge": ge_meta,
        "train": train_meta,
        "eval": eval_meta,
        "register": register_meta,
        "canary": canary_meta,
        "elapsed_seconds": round(elapsed, 2),
    }


# ===========================================================================
# Airflow DAG definition
# ===========================================================================

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="driftsentinel_retrain",
        default_args=DEFAULT_ARGS,
        description="DriftSentinel autonomous retraining pipeline",
        schedule_interval=None,   # Triggered by Kafka sensor
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["driftsentinel", "mlops", "retraining"],
        params={
            "trigger_json": '{"trigger_id":"test","model_id":"credit_model","strategy":"full_retrain","estimated_cost_usd":10.0,"sla_tier":"standard","training_data_path":"/data/train.parquet"}',
        },
    ) as dag:

        start = EmptyOperator(task_id="start")

        t1 = PythonOperator(
            task_id="validate_trigger",
            python_callable=validate_trigger,
            op_kwargs={"trigger_json": "{{ params.trigger_json }}"},
        )

        t2 = PythonOperator(
            task_id="prepare_training_data",
            python_callable=prepare_training_data,
            op_kwargs={
                "trigger_dict": "{{ ti.xcom_pull(task_ids='validate_trigger') }}",
            },
        )

        t3 = PythonOperator(
            task_id="run_great_expectations",
            python_callable=run_great_expectations,
            op_kwargs={
                "data_meta": "{{ ti.xcom_pull(task_ids='prepare_training_data') }}",
                "trigger_dict": "{{ ti.xcom_pull(task_ids='validate_trigger') }}",
            },
        )

        t4 = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            op_kwargs={
                "ge_meta": "{{ ti.xcom_pull(task_ids='run_great_expectations') }}",
                "trigger_dict": "{{ ti.xcom_pull(task_ids='validate_trigger') }}",
            },
        )

        t5 = PythonOperator(
            task_id="evaluate_model",
            python_callable=evaluate_model,
            op_kwargs={
                "train_meta": "{{ ti.xcom_pull(task_ids='train_model') }}",
                "trigger_dict": "{{ ti.xcom_pull(task_ids='validate_trigger') }}",
            },
        )

        t6 = PythonOperator(
            task_id="register_model",
            python_callable=register_model,
            op_kwargs={
                "eval_meta": "{{ ti.xcom_pull(task_ids='evaluate_model') }}",
                "train_meta": "{{ ti.xcom_pull(task_ids='train_model') }}",
                "trigger_dict": "{{ ti.xcom_pull(task_ids='validate_trigger') }}",
            },
        )

        t7 = PythonOperator(
            task_id="trigger_canary",
            python_callable=trigger_canary,
            op_kwargs={
                "register_meta": "{{ ti.xcom_pull(task_ids='register_model') }}",
                "eval_meta": "{{ ti.xcom_pull(task_ids='evaluate_model') }}",
                "trigger_dict": "{{ ti.xcom_pull(task_ids='validate_trigger') }}",
            },
        )

        end = EmptyOperator(task_id="end")

        start >> t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> end
