"""
Great Expectations Data Quality Gate

Validates incoming micro-batches BEFORE any drift test runs.
Bad data (nulls, type violations, impossible values) is quarantined
and does NOT pollute drift statistics.

Expectation suites are defined per model_id and versioned.
Suites are loaded from GCS/S3/local filesystem at startup and cached.

Quarantined records are emitted to the ge-quarantine Kafka topic.
Clean records pass through to the drift engine.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.core import ExpectationSuite
from great_expectations.data_context import AbstractDataContext

from src.ingestion.schema import (
    BatchFeatureStats,
    FeatureDistributionStats,
    GEValidationFailure,
)
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default expectation suite builder
# ---------------------------------------------------------------------------

def build_default_suite(
    model_id: str,
    reference_stats: list[FeatureDistributionStats],
) -> dict:
    """
    Auto-generate an expectation suite from reference statistics.
    Covers: null rates, value ranges, allowed categories, type consistency.
    Returns a dict that can be serialized to JSON for GE.
    """
    expectations = []

    for feat in reference_stats:
        col = feat.feature_name
        null_pct = feat.null_count / max(feat.total_count, 1)

        # Null rate expectation
        expectations.append({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": col,
                "mostly": 1.0 - min(null_pct * 2, 0.5),  # Allow 2x observed null rate
            },
        })

        # Range expectations for continuous features
        if feat.feature_type.value == "continuous" and feat.min is not None and feat.max is not None:
            slack = abs(feat.max - feat.min) * 0.5   # 50% slack on observed range
            expectations.append({
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": col,
                    "min_value": feat.min - slack,
                    "max_value": feat.max + slack,
                    "mostly": 0.95,
                },
            })

        # Categorical set expectation
        if feat.feature_type.value in ("categorical", "binary") and feat.value_counts:
            expected_values = list(feat.value_counts.keys())
            expectations.append({
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": col,
                    "value_set": expected_values,
                    "mostly": 0.98,
                },
            })

    return {
        "expectation_suite_name": f"{model_id}_auto_suite",
        "expectations": expectations,
        "meta": {
            "generated_by": "DriftSentinel",
            "model_id": model_id,
            "generated_at": datetime.utcnow().isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

@dataclass
class GEValidator:
    """
    Wraps Great Expectations validation for DriftSentinel.
    One instance per deployment; suites are cached per model_id.
    """

    suite_store_path: Path = field(
        default_factory=lambda: Path(settings.ge.suite_store_path)
    )
    _context: AbstractDataContext | None = field(default=None, init=False)
    _suite_cache: dict[str, dict] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.suite_store_path.mkdir(parents=True, exist_ok=True)
        self._context = gx.get_context(mode="ephemeral")

    # ------------------------------------------------------------------
    # Suite management
    # ------------------------------------------------------------------

    def register_suite(
        self, model_id: str, suite_dict: dict, overwrite: bool = False
    ) -> None:
        suite_path = self.suite_store_path / f"{model_id}.json"
        if suite_path.exists() and not overwrite:
            logger.info("suite_already_exists", model_id=model_id)
            self._suite_cache[model_id] = json.loads(suite_path.read_text())
            return
        suite_path.write_text(json.dumps(suite_dict, indent=2))
        self._suite_cache[model_id] = suite_dict
        logger.info("suite_registered", model_id=model_id, expectations=len(suite_dict.get("expectations", [])))

    def auto_register_from_reference(
        self,
        model_id: str,
        reference_stats: list[FeatureDistributionStats],
    ) -> None:
        suite_dict = build_default_suite(model_id, reference_stats)
        self.register_suite(model_id, suite_dict, overwrite=False)

    def _load_suite(self, model_id: str) -> dict | None:
        if model_id in self._suite_cache:
            return self._suite_cache[model_id]
        suite_path = self.suite_store_path / f"{model_id}.json"
        if suite_path.exists():
            suite = json.loads(suite_path.read_text())
            self._suite_cache[model_id] = suite
            return suite
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_batch(
        self,
        model_id: str,
        df: pd.DataFrame,
        batch_id: str,
    ) -> tuple[pd.DataFrame, GEValidationFailure | None]:
        """
        Validate DataFrame against model's expectation suite.

        Returns:
            (clean_df, failure_record)
            - clean_df: rows that passed validation
            - failure_record: GEValidationFailure if any expectations failed, else None
        """
        suite_dict = self._load_suite(model_id)
        if suite_dict is None:
            logger.warning("no_suite_skip_validation", model_id=model_id)
            return df, None

        failed_expectations: list[dict] = []
        quarantine_mask = pd.Series([False] * len(df), index=df.index)

        for expectation in suite_dict.get("expectations", []):
            exp_type = expectation["expectation_type"]
            kwargs = expectation["kwargs"]
            col = kwargs.get("column")
            mostly = kwargs.get("mostly", 1.0)

            if col not in df.columns:
                continue

            fail_mask = self._eval_expectation(df, exp_type, kwargs)
            fail_rate = fail_mask.sum() / max(len(df), 1)

            if fail_rate > (1.0 - mostly):
                failed_expectations.append({
                    "expectation_type": exp_type,
                    "column": col,
                    "fail_rate": float(fail_rate),
                    "mostly_threshold": mostly,
                })
                quarantine_mask |= fail_mask

        quarantined_count = int(quarantine_mask.sum())
        clean_df = df[~quarantine_mask]

        failure: GEValidationFailure | None = None
        if failed_expectations:
            failure = GEValidationFailure(
                batch_id=batch_id,
                model_id=model_id,
                expectation_suite=suite_dict.get("expectation_suite_name", "unknown"),
                failed_expectations=failed_expectations,
                quarantined_record_count=quarantined_count,
                total_record_count=len(df),
            )
            logger.warning(
                "ge_validation_failed",
                model_id=model_id,
                quarantined=quarantined_count,
                total=len(df),
                failed_expectations=len(failed_expectations),
            )
        else:
            logger.debug("ge_validation_passed", model_id=model_id, records=len(df))

        return clean_df, failure

    def _eval_expectation(
        self, df: pd.DataFrame, exp_type: str, kwargs: dict
    ) -> pd.Series:
        """Returns a boolean mask where True = row FAILED the expectation."""
        col = kwargs.get("column")
        if col is None or col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

        series = df[col]
        false_mask = pd.Series([False] * len(df), index=df.index)

        if exp_type == "expect_column_values_to_not_be_null":
            return series.isna()

        elif exp_type == "expect_column_values_to_be_between":
            numeric = pd.to_numeric(series, errors="coerce")
            min_v = kwargs.get("min_value")
            max_v = kwargs.get("max_value")
            fail = pd.Series([False] * len(df), index=df.index)
            if min_v is not None:
                fail |= numeric < min_v
            if max_v is not None:
                fail |= numeric > max_v
            return fail | numeric.isna()

        elif exp_type == "expect_column_values_to_be_in_set":
            value_set = set(kwargs.get("value_set", []))
            return ~series.isin(value_set) & series.notna()

        elif exp_type == "expect_column_values_to_match_regex":
            pattern = kwargs.get("regex", ".*")
            return ~series.astype(str).str.match(pattern)

        return false_mask
