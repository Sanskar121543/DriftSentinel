# DriftSentinel Makefile
# Usage: make <target>

.DEFAULT_GOAL := help
PYTHON        := python3
PIP           := pip3
COMPOSE       := docker compose

# ─── Help ──────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ─── Install ───────────────────────────────────────────────────────────────

.PHONY: install
install:  ## Install production dependencies
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev:  ## Install lightweight dev dependencies (no Spark/Airflow)
	$(PIP) install -r requirements-dev.txt

# ─── Infrastructure ────────────────────────────────────────────────────────

.PHONY: up
up:  ## Start all services (Kafka, MLflow, Airflow, Prometheus, Grafana)
	$(COMPOSE) up -d
	@echo ""
	@echo "  Services starting:"
	@echo "  Kafka UI  → http://localhost:8082"
	@echo "  Airflow   → http://localhost:8080  (admin/airflow)"
	@echo "  MLflow    → http://localhost:5000"
	@echo "  API       → http://localhost:8000/docs"
	@echo "  Prometheus→ http://localhost:9090"
	@echo "  Grafana   → http://localhost:3000  (admin/admin)"
	@echo ""

.PHONY: down
down:  ## Stop all services
	$(COMPOSE) down

.PHONY: down-volumes
down-volumes:  ## Stop services AND delete all volumes (destructive!)
	$(COMPOSE) down -v

.PHONY: logs
logs:  ## Follow all service logs
	$(COMPOSE) logs -f

.PHONY: logs-api
logs-api:  ## Follow API logs only
	$(COMPOSE) logs -f api

.PHONY: setup-topics
setup-topics:  ## Create Kafka topics (run after `make up`)
	$(COMPOSE) exec kafka bash /scripts/setup_topics.sh

.PHONY: api-dev
api-dev:  ## Run API locally (no Docker)
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# ─── Tests ─────────────────────────────────────────────────────────────────

.PHONY: test
test:  ## Run all unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov:  ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ─── Benchmarks ────────────────────────────────────────────────────────────

.PHONY: benchmark
benchmark:  ## Run ALL benchmarks and print pass/fail summary
	$(PYTHON) -m benchmarks.run_all

.PHONY: benchmark-fast
benchmark-fast:  ## Fast benchmark (2 trials per condition, for CI)
	$(PYTHON) -m benchmarks.run_all --fast

.PHONY: benchmark-mttd
benchmark-mttd:  ## MTTD benchmark only (drift injection, ~3.8h target)
	$(PYTHON) -m benchmarks.drift_injection_benchmark 5

.PHONY: benchmark-strategy
benchmark-strategy:  ## Strategy selector benchmark only (94% accuracy target)
	$(PYTHON) -m benchmarks.strategy_eval_benchmark

.PHONY: demo
demo:  ## Run standalone end-to-end demo (no Docker needed)
	$(PYTHON) demo.py

# ─── Build ─────────────────────────────────────────────────────────────────

.PHONY: build
build:  ## Build Docker images
	docker build --target api    -t driftsentinel-api:latest .
	docker build --target worker -t driftsentinel-worker:latest .

.PHONY: build-spark
build-spark:  ## Build Spark feature aggregator image
	docker build --target spark -t driftsentinel-spark:latest .

# ─── Lint / Format ─────────────────────────────────────────────────────────

.PHONY: lint
lint:  ## Run ruff linter
	ruff check src/ benchmarks/ tests/

.PHONY: fmt
fmt:  ## Run ruff formatter
	ruff format src/ benchmarks/ tests/

.PHONY: typecheck
typecheck:  ## Run mypy type checker
	mypy src/ --ignore-missing-imports --no-strict-optional

# ─── Cleanup ───────────────────────────────────────────────────────────────

.PHONY: clean
clean:  ## Remove cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf benchmarks/results/

.PHONY: clean-models
clean-models:  ## Remove generated model files
	rm -f models/strategy_selector.pkl models/strategy_selector.rules.txt
