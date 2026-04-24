"""
Centralized configuration using Pydantic Settings.
Values are loaded from environment variables with .env file fallback.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_")

    bootstrap_servers: str = "localhost:9092"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None


class SparkSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPARK_")

    master: str = "local[*]"
    app_name: str = "DriftSentinel"
    checkpoint_path: str = "/tmp/driftsentinel/checkpoints"
    executor_memory: str = "4g"
    driver_memory: str = "2g"
    executor_cores: int = 2
    shuffle_partitions: int = 50


class DriftSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DRIFT_")

    window_minutes: int = 5
    watermark_minutes: int = 10
    min_alert_tests: int = 2
    ks_pvalue_threshold: float = 0.05
    chi2_pvalue_threshold: float = 0.05
    psi_threshold: float = 0.2
    js_threshold: float = 0.1
    shap_delta_threshold: float = 0.15


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    api_key: str = ""
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 2000
    temperature: float = 0.1


class PineconeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PINECONE_")

    api_key: str = ""
    environment: str = "us-east-1-aws"
    cloud: str = "aws"
    region: str = "us-east-1"


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "driftsentinel"
    artifact_location: str = "gs://driftsentinel-artifacts/mlflow"


class RetrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRAINING_")

    cost_ceiling_usd: float = 50.0
    selector_model_path: str = "models/strategy_selector.pkl"
    weighted_lambda: float = 0.05   # Exponential decay for weighted retrain


class CanarySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CANARY_")

    stages: list[float] = [0.05, 0.20, 0.50, 1.00]
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.10
    sprt_mde: float = 0.02
    max_stage_hours: int = 24
    min_stage_samples: int = 500


class GESettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GE_")

    suite_store_path: str = "configs/ge_suites"


class K8sSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="K8S_")

    namespace: str = "driftsentinel"
    in_cluster: bool = False


class JiraSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JIRA_")

    base_url: str = ""
    project_key: str = "DS"
    email: str = ""
    api_token: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    spark: SparkSettings = Field(default_factory=SparkSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    retraining: RetrainingSettings = Field(default_factory=RetrainingSettings)
    canary: CanarySettings = Field(default_factory=CanarySettings)
    ge: GESettings = Field(default_factory=GESettings)
    k8s: K8sSettings = Field(default_factory=K8sSettings)
    jira: JiraSettings = Field(default_factory=JiraSettings)

    environment: str = "development"
    log_level: str = "INFO"
    log_format: str = "json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
