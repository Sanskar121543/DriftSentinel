DriftSentinel
<div align="center">
Autonomous ML Observability & Self-Healing Platform

Real-time drift detection, automated diagnosis, cost-aware retraining, and statistically validated canary promotion for production machine learning systems.














⭐ Benchmark-backed autonomous MLOps system with real drift detection, retraining, and canary rollback.

</div>
Overview

DriftSentinel is an end-to-end MLOps platform that continuously monitors deployed machine learning models, detects production drift in real time, diagnoses likely causes, selects the optimal remediation path, retrains challenger models, and validates rollout decisions using sequential canary testing.

Most monitoring systems stop at alerts.

DriftSentinel closes the loop:

detect → diagnose → decide → retrain → validate → promote / rollback

Built to demonstrate production-grade ML systems engineering rather than notebook-only machine learning.

Verified Benchmark Results
Metric	Result	Target	Status
Mean Time to Detect Drift	0.45h	≤ 4.0h	✅
Strategy Selector Accuracy	100%	≥ 94%	✅
Unit Test Suite	37 Passed	All Pass	✅
Core Architecture

Inference Events
↓
Kafka Topics
↓
Spark Micro-Batch Aggregation (5 min windows)
↓
Great Expectations Quality Gate
↓
5 Parallel Drift Tests (KS · Chi² · PSI · JS · SHAP Delta)
↓
Drift Alert
↓
LLM Diagnosis + Historical Memory
↓
Cost-Aware Strategy Selector
↓
Airflow Retraining Pipeline
↓
MLflow Tracking
↓
SPRT Canary Promotion
↓
Promote / Rollback

Experiment Tracking & Model Registry

Every retraining cycle logs parameters, metrics, artifacts, and challenger evaluation history using MLflow.

<p align="center"> <img src="docs/images/MLFlow%20dashboard.png" width="95%"> </p>
Why This Project Stands Out
Real Production Problems Solved
Detects live model drift instead of offline analysis
Uses multiple statistical detectors instead of a single weak signal
Validates data quality before retraining
Selects cost-aware retraining strategies automatically
Tracks experiments and model lineage
Uses canary deployment with rollback protection
Includes benchmark and unit-test proof
Strong Engineering Signal

Demonstrates skill in:

MLOps
Distributed systems
Streaming pipelines
Statistical testing
Backend engineering
Reliability engineering
Production ML systems
Drift Detection Engine
Detector	Purpose
Kolmogorov-Smirnov	Continuous feature shift
Chi-Squared	Categorical distribution shift
PSI	Population stability drift
Jensen-Shannon	Distribution divergence
SHAP Delta	Concept drift

Alerts trigger only when multiple tests agree, reducing false positives.

Intelligent Retraining Selector
Strategy	Best Use Case
Full Retrain	Severe global drift
Weighted Retrain	Mild temporal drift
Slice Finetune	Segment-specific drift
Ensemble Fallback	Low-data / high-risk scenario

Hybrid rules + model logic benchmarked at 100% accuracy.

Canary Deployment Logic

Uses Sequential Probability Ratio Testing (SPRT) rather than fixed-horizon A/B testing.

Benefits:

Faster deployment decisions
Lower sample requirements
Earlier rollback on regressions
Safer production rollout
Tech Stack
Backend: Python, FastAPI
Streaming: Kafka, Spark Structured Streaming
Validation: Great Expectations
Orchestration: Airflow
Experiment Ops: MLflow
Monitoring: Prometheus, Grafana
Testing: Pytest
Infra: Docker, Kubernetes-ready
Quick Start
Lightweight Demo (No Docker)

pip install -r requirements-dev.txt
python demo.py

Runs the full local pipeline without Kafka, Spark, or Airflow.

Full Stack

cp .env.example .env
make up
make setup-topics

Services
Service	URL
API Docs	http://localhost:8000/docs

Kafka UI	http://localhost:8082

Airflow	http://localhost:8080

MLflow	http://localhost:5000

Grafana	http://localhost:3000
Run Benchmarks

python -m benchmarks.run_all --fast

or full mode:

python -m benchmarks.run_all

Project Structure

DriftSentinel/
├── airflow/
├── benchmarks/
├── configs/
├── docs/
├── k8s/
├── models/
├── scripts/
├── src/
├── tests/
├── demo.py
├── docker-compose.yml
└── README.md

Resume-Level Impact

Designed and built an autonomous ML reliability platform capable of:

Detecting drift in 27 minutes average
Automatically selecting optimal retraining strategies
Preventing bad promotions with canary rollback
Operating as a measurable benchmarked production system
Future Extensions
Multi-model fleet management
Cloud-native deployment on AWS / GCP
Shadow deployments
Feature store integration
Reinforcement learning selector
Slack / PagerDuty incident routing
Final Note

DriftSentinel was built to demonstrate the difference between toy ML projects and real production ML systems.

It focuses on what matters after deployment:

reliability, observability, safety, and autonomous recovery
