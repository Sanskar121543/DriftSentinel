<div align="center">

<br/>

```
██████╗ ██████╗ ██╗███████╗████████╗    ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝    ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
██║  ██║██████╔╝██║█████╗     ██║       ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
██║  ██║██╔══██╗██║██╔══╝     ██║       ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
██████╔╝██║  ██║██║██║        ██║       ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝        ╚═╝       ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

### Autonomous ML Observability & Self-Healing Platform

*Real-time drift detection · Automated diagnosis · Cost-aware retraining · Statistically validated canary promotion*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-Streaming-231F20?style=flat-square&logo=apache-kafka&logoColor=white)](https://kafka.apache.org)
[![Spark](https://img.shields.io/badge/Apache_Spark-Micro--Batch-E25A1C?style=flat-square&logo=apache-spark&logoColor=white)](https://spark.apache.org)
[![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?style=flat-square&logo=apache-airflow&logoColor=white)](https://airflow.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-37_Passed-22C55E?style=flat-square&logo=pytest&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square)](LICENSE)

<br/>

> **Most monitoring systems stop at alerts.**
> DriftSentinel closes the loop.

<br/>

```
DETECT → DIAGNOSE → DECIDE → RETRAIN → VALIDATE → PROMOTE / ROLLBACK
```

</div>

---

## Benchmark Results

> All results are reproducible. Run `python -m benchmarks.run_all` to verify.

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Mean Time to Detect Drift | **0.45 hours** | ≤ 4.0h | ✅ **9× faster than target** |
| Strategy Selector Accuracy | **100%** | ≥ 94% | ✅ **Perfect** |
| Unit Test Suite | **37 / 37 Passed** | All Pass | ✅ **Clean** |

---

## Why DriftSentinel

The gap between a Jupyter notebook and a production ML system is enormous. Most ML projects demonstrate training accuracy. DriftSentinel demonstrates what matters after deployment: **reliability, observability, safety, and autonomous recovery**.

| What most projects do | What DriftSentinel does |
|---|---|
| Offline drift analysis | Live drift detection on production traffic |
| Single statistical test | 5 parallel detectors with consensus logic |
| Manual retraining | Automated, cost-aware strategy selection |
| Fixed A/B test windows | Sequential canary testing with early stopping |
| Ad-hoc experiments | Full MLflow lineage and model registry |
| No rollback logic | SPRT-gated promotion with automatic rollback |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRODUCTION INFERENCE                       │
│                     (Real-time Events)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       KAFKA TOPICS                              │
│              inference-events · feature-logs                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           SPARK STRUCTURED STREAMING  (5-min windows)           │
│           Micro-batch aggregation · Feature statistics          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               GREAT EXPECTATIONS  Quality Gate                  │
│           Schema validation · Null checks · Range guards        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │   DRIFT DETECTION     │  5 tests run in parallel
              │                       │
              │  KS    · Chi²         │  Continuous features
              │  PSI   · JS           │  Distribution stability
              │  SHAP Delta           │  Concept drift
              │                       │
              │  Alert if ≥ 2 agree   │  Multi-signal consensus
              └───────────┬───────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LLM DIAGNOSIS ENGINE                           │
│         Root cause inference · Historical pattern memory        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              COST-AWARE STRATEGY SELECTOR                       │
│     Full Retrain · Weighted · Slice Finetune · Ensemble         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           AIRFLOW RETRAINING PIPELINE  +  MLFLOW               │
│      Challenger training · Artifact logging · Lineage           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           SPRT CANARY DEPLOYMENT  (Sequential Testing)          │
│     Faster decisions · Lower sample cost · Early rollback       │
└──────────┬──────────────────────────────────────┬──────────────┘
           │                                      │
           ▼                                      ▼
    ✅  PROMOTE                            ❌  ROLLBACK
```

---

## Core Components

### Drift Detection Engine

Five detectors run simultaneously. An alert fires only when multiple agree — dramatically reducing false positives.

| Detector | Catches |
|----------|---------|
| **Kolmogorov-Smirnov** | Continuous feature distribution shift |
| **Chi-Squared** | Categorical feature distribution shift |
| **PSI** | Population stability over time |
| **Jensen-Shannon** | Divergence between reference and production |
| **SHAP Delta** | Concept drift — input/output relationship breaks |

### Intelligent Strategy Selector

Hybrid rules + ML logic, benchmarked at **100% accuracy** across all test scenarios.

| Strategy | Triggers When |
|----------|--------------|
| **Full Retrain** | Severe global drift across all features |
| **Weighted Retrain** | Mild temporal drift — recent data upweighted |
| **Slice Finetune** | Drift isolated to a specific segment or cohort |
| **Ensemble Fallback** | Low data volume or elevated deployment risk |

### Canary Deployment with SPRT

DriftSentinel uses **Sequential Probability Ratio Testing** rather than fixed-horizon A/B testing.

- ⚡ Reaches decisions faster with less data
- 🛡️ Detects regressions earlier
- 💰 Lower sample cost per rollout
- 🔁 Automatic rollback if challenger underperforms

---

## Experiment Tracking

Every retraining cycle is fully logged in **MLflow**: parameters, metrics, artifacts, and the full challenger evaluation history.

![MLflow Dashboard](docs/images/MLFlow%20dashboard.png)

All model versions are registered with lineage back to the drift event that triggered retraining.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | Python · FastAPI |
| **Streaming** | Apache Kafka · Spark Structured Streaming |
| **Data Quality** | Great Expectations |
| **Orchestration** | Apache Airflow |
| **Experiment Tracking** | MLflow |
| **Observability** | Prometheus · Grafana |
| **Testing** | Pytest (37 tests) |
| **Infra** | Docker · Kubernetes-ready |

---

## Quick Start

### Lightweight Demo *(no Docker required)*

```bash
pip install -r requirements-dev.txt
python demo.py
```

Runs the complete local pipeline — drift detection, diagnosis, strategy selection, and canary logic — without Kafka, Spark, or Airflow.

### Full Stack

```bash
cp .env.example .env
make up
make setup-topics
```

| Service | URL |
|---------|-----|
| API Docs | http://localhost:8000/docs |
| Kafka UI | http://localhost:8082 |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |

### Run Benchmarks

```bash
# Fast mode
python -m benchmarks.run_all --fast

# Full benchmark suite
python -m benchmarks.run_all
```

---

## Project Structure

```
DriftSentinel/
├── airflow/          # DAG definitions for retraining pipelines
├── benchmarks/       # Reproducible benchmark suite
├── configs/          # Environment and pipeline configuration
├── docs/             # Architecture diagrams and images
├── k8s/              # Kubernetes manifests
├── models/           # Model artifacts and registry
├── scripts/          # Utility and setup scripts
├── src/
│   ├── detection/    # KS, Chi², PSI, JS, SHAP detectors
│   ├── diagnosis/    # LLM diagnosis engine + memory
│   ├── selector/     # Cost-aware strategy selector
│   ├── retraining/   # Airflow pipeline + MLflow integration
│   └── canary/       # SPRT-based canary deployment
├── tests/            # 37-test pytest suite
├── demo.py           # Lightweight local demo
└── docker-compose.yml
```

---

## Roadmap

- [ ] Multi-model fleet management
- [ ] Cloud-native deployment on AWS / GCP
- [ ] Shadow mode deployments
- [ ] Feature store integration
- [ ] RL-based strategy selector
- [ ] Slack / PagerDuty incident routing

---

<div align="center">

**DriftSentinel was built to demonstrate the difference between toy ML projects and production ML systems.**

*The hard part of machine learning isn't training a model. It's keeping it working.*

<br/>

[![Star this repo](https://img.shields.io/github/stars/yourusername/DriftSentinel?style=social)](https://github.com/yourusername/DriftSentinel)

</div>
