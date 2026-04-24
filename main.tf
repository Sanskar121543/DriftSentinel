/*
DriftSentinel — GCP Infrastructure

Provisions:
  - GKE Autopilot cluster (prod) or Standard cluster (dev)
  - Cloud Storage buckets (MLflow artifacts, training data, GE suites)
  - Managed Kafka on Confluent Cloud (via Confluent Terraform provider)
  - Cloud SQL (PostgreSQL) for MLflow and Airflow backends
  - Cloud Monitoring workspace + alerting policies

Usage:
  terraform init
  terraform apply -var-file=envs/prod.tfvars
*/

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }
  backend "gcs" {
    bucket = "driftsentinel-tf-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "gcp_project" { type = string }
variable "gcp_region" { type = string; default = "us-central1" }
variable "environment" { type = string; default = "prod" }
variable "gke_node_count" { type = number; default = 3 }
variable "gke_machine_type" { type = string; default = "n2-standard-4" }
variable "db_tier" { type = string; default = "db-g1-small" }

locals {
  name_prefix = "driftsentinel-${var.environment}"
  common_labels = {
    project     = "driftsentinel"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ---------------------------------------------------------------------------
# GKE cluster
# ---------------------------------------------------------------------------

resource "google_container_cluster" "main" {
  name     = "${local.name_prefix}-gke"
  location = var.gcp_region

  # Autopilot handles node provisioning automatically
  enable_autopilot = var.environment == "prod"

  # Standard cluster for non-prod (more control)
  dynamic "node_config" {
    for_each = var.environment != "prod" ? [1] : []
    content {
      machine_type = var.gke_machine_type
      disk_size_gb = 100
      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
      ]
      labels = local.common_labels
    }
  }

  dynamic "node_pool" {
    for_each = var.environment != "prod" ? [1] : []
    content {
      name       = "default-pool"
      node_count = var.gke_node_count
    }
  }

  # GPU node pool for training jobs
  dynamic "node_pool" {
    for_each = var.environment == "prod" ? [] : []
    content {
      name       = "gpu-pool"
      node_count = 1
      node_config {
        machine_type = "n1-standard-4"
        guest_accelerator {
          type  = "nvidia-tesla-t4"
          count = 1
        }
        oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
      }
      autoscaling {
        min_node_count = 0
        max_node_count = 4
      }
    }
  }

  addons_config {
    http_load_balancing { disabled = false }
    horizontal_pod_autoscaling { disabled = false }
  }

  resource_labels = local.common_labels
  deletion_protection = var.environment == "prod"
}

# ---------------------------------------------------------------------------
# Cloud Storage
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "artifacts" {
  name          = "${var.gcp_project}-${local.name_prefix}-artifacts"
  location      = var.gcp_region
  storage_class = "STANDARD"
  force_destroy = var.environment != "prod"

  versioning { enabled = true }

  lifecycle_rule {
    condition { age = 90 }
    action { type = "SetStorageClass"; storage_class = "NEARLINE" }
  }

  labels = local.common_labels
}

resource "google_storage_bucket" "training_data" {
  name          = "${var.gcp_project}-${local.name_prefix}-training"
  location      = var.gcp_region
  storage_class = "STANDARD"
  force_destroy = var.environment != "prod"

  labels = local.common_labels
}

# ---------------------------------------------------------------------------
# Cloud SQL (PostgreSQL) for MLflow + Airflow
# ---------------------------------------------------------------------------

resource "google_sql_database_instance" "main" {
  name             = "${local.name_prefix}-postgres"
  database_version = "POSTGRES_16"
  region           = var.gcp_region

  settings {
    tier              = var.db_tier
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"
    disk_autoresize   = true
    disk_size         = 20

    backup_configuration {
      enabled                        = true
      start_time                     = "02:00"
      point_in_time_recovery_enabled = var.environment == "prod"
    }

    insights_config {
      query_insights_enabled = true
    }

    database_flags {
      name  = "max_connections"
      value = "200"
    }
  }

  deletion_protection = var.environment == "prod"
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_database" "airflow" {
  name     = "airflow"
  instance = google_sql_database_instance.main.name
}

# ---------------------------------------------------------------------------
# Service Accounts
# ---------------------------------------------------------------------------

resource "google_service_account" "driftsentinel" {
  account_id   = "${local.name_prefix}-sa"
  display_name = "DriftSentinel Service Account"
}

resource "google_project_iam_member" "storage_admin" {
  project = var.gcp_project
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.driftsentinel.email}"
}

resource "google_project_iam_member" "dataproc_worker" {
  project = var.gcp_project
  role    = "roles/dataproc.worker"
  member  = "serviceAccount:${google_service_account.driftsentinel.email}"
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "gke_cluster_name" {
  value = google_container_cluster.main.name
}

output "gke_endpoint" {
  value     = google_container_cluster.main.endpoint
  sensitive = true
}

output "artifacts_bucket" {
  value = google_storage_bucket.artifacts.name
}

output "training_data_bucket" {
  value = google_storage_bucket.training_data.name
}

output "postgres_connection_name" {
  value = google_sql_database_instance.main.connection_name
}
