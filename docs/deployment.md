# Deployment Guide

## Local Development

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Docker | 24+ | https://docs.docker.com/get-docker/ |
| Docker Compose | v2 | bundled with Docker Desktop |
| Python | 3.11+ | https://python.org |
| Make | any | bundled on macOS/Linux |

**RAM:** 16GB minimum (Kafka + Spark + MLflow + Airflow = heavy stack)

### First-time setup

```bash
git clone https://github.com/yourusername/driftsentinel.git
cd driftsentinel

# Install Python deps for running scripts locally
pip install -r requirements.txt -r requirements-dev.txt

# Copy and fill env vars
cp .env.example .env
# Edit .env — at minimum set:
#   LLM_API_KEY       (OpenAI key with GPT-4o access)
#   PINECONE_API_KEY  (Pinecone account)

# Start all services
make up

# Wait ~60 seconds for services to be healthy, then:
make seed          # Create Kafka topics + seed 3 demo model references
make smoke         # Verify everything works end-to-end (~30 seconds)
```

### Running the drift detection loop

```bash
# Terminal 1: Spark feature aggregator (reads inference-events → feature-stats)
make run-spark

# Terminal 2: Drift detection engine (reads feature-stats → drift-alerts)
make run-detector

# Terminal 3: LLM diagnosis worker (reads drift-alerts → diagnosis reports)
make run-diagnosis  # Requires OPENAI_API_KEY

# Terminal 4: Inject synthetic inference events
make simulate
```

### Injecting drift manually

```bash
# Inject drift into the credit-risk-v2 model for a quick demo
python scripts/inject_drift.py --model credit-risk-v2 --severity high --type covariate
```

### Running the MTTD benchmark

```bash
make benchmark
# Results appear in benchmarks/results/
#   benchmark_raw.csv     — one row per trial
#   benchmark_summary.csv — mean/p50/p95 MTTD per drift type × severity
#   detection_rates.csv   — % of trials where drift was detected
```

---

## GCP Deployment

### Prerequisites

```bash
# Install tools
brew install google-cloud-sdk terraform helm kubectl

# Authenticate
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 1. Provision infrastructure

```bash
cd terraform

# Initialize providers
terraform init

# Preview changes (check before applying)
terraform plan -var-file=envs/prod.tfvars

# Apply (~10 minutes)
terraform apply -var-file=envs/prod.tfvars
```

**`envs/prod.tfvars`:**

```hcl
gcp_project      = "your-project-id"
gcp_region       = "us-central1"
environment      = "prod"
gke_node_count   = 3
gke_machine_type = "n2-standard-8"
db_tier          = "db-n1-standard-2"
```

### 2. Build and push images

```bash
# Configure Docker for GCR
gcloud auth configure-docker gcr.io

# Build + push (tags with git commit SHA)
make docker-push REGISTRY=gcr.io/your-project-id
```

### 3. Configure GKE credentials

```bash
gcloud container clusters get-credentials driftsentinel-prod-gke \
  --region us-central1 \
  --project your-project-id
```

### 4. Create secrets

```bash
kubectl create namespace driftsentinel

kubectl create secret generic driftsentinel-secrets \
  --namespace driftsentinel \
  --from-literal=LLM_API_KEY="sk-..." \
  --from-literal=PINECONE_API_KEY="..." \
  --from-literal=JIRA_API_TOKEN="..." \
  --from-literal=MLFLOW_TRACKING_URI="http://mlflow-service:5000"
```

### 5. Deploy to GKE

```bash
make k8s-deploy ENV=prod

# Verify pods are running
kubectl get pods -n driftsentinel

# Check API health
kubectl port-forward svc/driftsentinel-api 8000:80 -n driftsentinel
curl http://localhost:8000/health
```

### 6. Deploy Airflow on GKE (Helm)

```bash
helm repo add apache-airflow https://airflow.apache.org
helm repo update

helm upgrade --install airflow apache-airflow/airflow \
  --namespace driftsentinel \
  --set executor=KubernetesExecutor \
  --set dags.persistence.enabled=true \
  --set postgresql.enabled=false \
  --set data.metadataConnection.host=CLOUD_SQL_HOST \
  --values k8s/airflow-values.yaml
```

### 7. Grafana dashboard

After deploying, port-forward Grafana and import the dashboard:

```bash
kubectl port-forward svc/grafana 3000:80 -n monitoring

# Open http://localhost:3000
# Import: configs/grafana/dashboards/driftsentinel.json
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEY` | Yes (diagnosis) | — | OpenAI API key |
| `PINECONE_API_KEY` | Yes (RAG) | — | Pinecone API key |
| `KAFKA_BOOTSTRAP_SERVERS` | Yes | `localhost:9092` | Kafka broker address |
| `MLFLOW_TRACKING_URI` | Yes | `http://localhost:5000` | MLflow server |
| `DRIFT_MIN_ALERT_TESTS` | No | `2` | Min tests that must agree to fire alert |
| `DRIFT_PSI_THRESHOLD` | No | `0.2` | PSI alert threshold |
| `RETRAINING_COST_CEILING_USD` | No | `50.0` | Max auto-retrain cost before escalating |
| `CANARY_SPRT_MDE` | No | `0.02` | Min detectable effect for canary SPRT |
| `JIRA_BASE_URL` | No | — | Jira instance URL for rollback tickets |

---

## Monitoring

### Key Dashboards

After `make up`, open Grafana at http://localhost:3000:

- **Drift Overview** — Alert rate, severity distribution, MTTD trend
- **Model Health** — Per-model drift status, drifted feature count
- **Canary Promotions** — Active canaries, SPRT LLR, stage progress
- **Infrastructure** — Kafka consumer lag, API latency p50/p99, pod count

### Key Prometheus Metrics

```
driftsentinel_alerts_total{model_id, severity}
driftsentinel_detection_duration_seconds{model_id}
driftsentinel_features_drifted{model_id}
driftsentinel_canary_stage_traffic_pct{model_id, challenger_version}
driftsentinel_models_monitored_total
driftsentinel_api_requests_total{method, path, status_code}
```

---

## Troubleshooting

**Kafka not starting:**
```bash
docker compose logs kafka
# Ensure port 9092 is not in use: lsof -i :9092
```

**MLflow not accessible:**
```bash
docker compose logs mlflow postgres
# Postgres must be healthy before MLflow starts
```

**Spark job fails with Java errors:**
```bash
# Ensure JAVA_HOME is set, or use the Docker target
make run-spark  # Uses local Spark if installed
# Or: docker compose exec api python -m src.streaming.feature_aggregator
```

**Drift not detected in benchmark:**
```bash
# Check reference was loaded
curl http://localhost:8000/models
# If empty, run: make seed
```

**LLM diagnosis not running:**
```bash
# LLM_API_KEY must be set in .env
# Diagnosis is async — alerts will still fire without it
# Check: docker compose logs api | grep diagnosis
```
