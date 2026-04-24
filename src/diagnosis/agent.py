"""
Multimodal LLM Diagnosis Agent

5-step structured reasoning chain:
  1. Receive the statistical anomaly report as structured JSON
  2. Retrieve 3 most similar past drift incidents from Pinecone via semantic search
  3. Render SHAP waterfall plots as PNG images → feed to vision LLM
  4. Generate ranked hypotheses with confidence scores
  5. Output structured DiagnosisReport: segments, root cause, strategy, business impact

Design:
  - Fully async: diagnosis runs on a separate worker pool, never blocking detection
  - LangChain LCEL pipeline for composability and streaming
  - OpenAI GPT-4o (vision) as backbone; swappable via config
  - Pinecone for incident embeddings + retrieval
  - MLflow for logging each diagnosis as a run artifact
"""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Any

import mlflow
import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from src.diagnosis.rag import IncidentRAG
from src.diagnosis.shap_renderer import render_shap_waterfall
from src.ingestion.schema import (
    DriftAlert,
    DiagnosisReport,
    RetrainingStrategy,
)
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are DriftSentinel's ML observability expert.
You receive statistical drift reports from production ML models and generate
precise root-cause analyses. You have access to similar past incidents
retrieved from a vector database.

ALWAYS respond with valid JSON matching this schema:
{
  "hypotheses": [
    {
      "hypothesis": "string — specific, falsifiable root cause",
      "confidence": 0.0–1.0,
      "evidence": ["list", "of", "supporting", "signals"]
    }
  ],
  "top_hypothesis": "string",
  "top_hypothesis_confidence": 0.0–1.0,
  "affected_segments": [{"key": "value"}],
  "recommended_strategy": "full_retrain|weighted_retrain|slice_finetune|ensemble_fallback",
  "strategy_rationale": "string",
  "estimated_impact_usd": float,
  "full_report_markdown": "string — detailed markdown report"
}

Ground your hypotheses in the statistical evidence provided.
Be specific: name the drifted features, quantify the drift magnitude,
and explain the likely upstream cause (data pipeline change, seasonal shift,
population shift, label drift, etc.).
When SHAP plots are provided, reference the specific feature importance changes.
When past incidents are provided, note similarities and differences explicitly."""


class DiagnosisAgent:
    """
    Orchestrates the 5-step LLM diagnosis pipeline.
    Thread-safe; one instance per worker process.
    """

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=0.1,
            max_tokens=2000,
            api_key=settings.llm.api_key,
        )
        self.rag = IncidentRAG()
        self.output_parser = JsonOutputParser()
        logger.info("diagnosis_agent_initialized", model=settings.llm.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def diagnose(
        self,
        alert: DriftAlert,
        shap_plot_dir: Path | None = None,
    ) -> DiagnosisReport:
        """Run the full 5-step diagnosis pipeline for a drift alert."""

        logger.info(
            "diagnosis_start",
            alert_id=alert.alert_id,
            model_id=alert.model_id,
            severity=alert.severity.value,
        )

        with mlflow.start_run(run_name=f"diagnosis-{alert.alert_id[:8]}"):
            mlflow.set_tags({
                "alert_id": alert.alert_id,
                "model_id": alert.model_id,
                "severity": alert.severity.value,
            })

            # Step 1: Structure the anomaly report
            anomaly_json = self._format_anomaly_report(alert)

            # Step 2: RAG — retrieve similar past incidents
            similar_incidents = await self.rag.retrieve_similar(
                alert=alert, top_k=3
            )

            # Step 3: Render SHAP plots
            shap_image_b64s: list[str] = []
            shap_plot_paths: list[str] = []
            if shap_plot_dir:
                for feat in alert.drifted_features[:4]:  # max 4 plots
                    path = shap_plot_dir / f"{alert.model_id}_{feat}.png"
                    if path.exists():
                        b64 = _encode_image_b64(path)
                        shap_image_b64s.append(b64)
                        shap_plot_paths.append(str(path))
                        mlflow.log_artifact(str(path))

            # Step 4 + 5: LLM call
            raw_output = await self._call_llm(
                anomaly_json, similar_incidents, shap_image_b64s
            )
            parsed = self._parse_output(raw_output, alert)

            # Build DiagnosisReport
            report = DiagnosisReport(
                diagnosis_id=str(uuid.uuid4()),
                alert_id=alert.alert_id,
                model_id=alert.model_id,
                similar_incidents=similar_incidents,
                hypotheses=parsed["hypotheses"],
                top_hypothesis=parsed["top_hypothesis"],
                top_hypothesis_confidence=parsed["top_hypothesis_confidence"],
                affected_segments=parsed.get("affected_segments", []),
                recommended_strategy=RetrainingStrategy(parsed["recommended_strategy"]),
                strategy_rationale=parsed["strategy_rationale"],
                estimated_impact_usd=parsed.get("estimated_impact_usd", 0.0),
                full_report_markdown=parsed["full_report_markdown"],
                shap_plot_paths=shap_plot_paths,
            )

            # Log to MLflow
            mlflow.log_metrics({
                "top_hypothesis_confidence": report.top_hypothesis_confidence,
                "estimated_impact_usd": report.estimated_impact_usd,
                "similar_incidents_found": len(similar_incidents),
            })
            mlflow.log_dict(
                report.model_dump(exclude={"embedding"}),
                "diagnosis_report.json",
            )

            # Index this incident for future RAG retrieval
            await self.rag.index_incident(report)

            logger.info(
                "diagnosis_complete",
                diagnosis_id=report.diagnosis_id,
                strategy=report.recommended_strategy.value,
                confidence=report.top_hypothesis_confidence,
            )
            return report

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _format_anomaly_report(self, alert: DriftAlert) -> str:
        test_summary = []
        for r in alert.test_results:
            if r.drifted:
                test_summary.append({
                    "test": r.test_name,
                    "feature": r.feature_name,
                    "statistic": round(r.statistic, 4),
                    "p_value": round(r.p_value, 4) if r.p_value else None,
                    "threshold": r.threshold,
                    "details": r.details,
                })

        report = {
            "alert_id": alert.alert_id,
            "model_id": alert.model_id,
            "severity": alert.severity.value,
            "window": {
                "start": alert.window_start.isoformat(),
                "end": alert.window_end.isoformat(),
            },
            "segment": alert.segment,
            "drifted_features": alert.drifted_features,
            "tests_fired": alert.tests_fired,
            "tests_total": alert.tests_total,
            "test_results": test_summary,
        }
        return json.dumps(report, indent=2)

    async def _call_llm(
        self,
        anomaly_json: str,
        similar_incidents: list[dict],
        shap_image_b64s: list[str],
    ) -> str:
        content: list[dict] = []

        # Text content
        user_text = f"""
## Current Drift Alert

```json
{anomaly_json}
```

## Similar Past Incidents (from incident history)

```json
{json.dumps(similar_incidents, indent=2)}
```

Based on the above, provide a complete root-cause diagnosis.
"""
        content.append({"type": "text", "text": user_text})

        # Vision content: SHAP waterfall plots
        for i, b64 in enumerate(shap_image_b64s):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            })
            content.append({
                "type": "text",
                "text": f"SHAP waterfall plot {i + 1} for drifted feature. Analyze the feature importance changes shown.",
            })

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    def _parse_output(self, raw: str, alert: DriftAlert) -> dict:
        """Parse LLM JSON output with fallback on parse failure."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "llm_parse_failed",
                alert_id=alert.alert_id,
                raw_length=len(raw),
            )
            return _fallback_diagnosis(alert)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _fallback_diagnosis(alert: DriftAlert) -> dict:
    """Safe fallback when LLM output is unparseable."""
    return {
        "hypotheses": [
            {
                "hypothesis": f"Statistical drift detected in {len(alert.drifted_features)} features. Manual review required.",
                "confidence": 0.5,
                "evidence": [f"{r.test_name} fired on {r.feature_name}" for r in alert.test_results if r.drifted],
            }
        ],
        "top_hypothesis": f"Drift in features: {', '.join(alert.drifted_features[:3])}",
        "top_hypothesis_confidence": 0.5,
        "affected_segments": [alert.segment] if alert.segment else [],
        "recommended_strategy": "full_retrain",
        "strategy_rationale": "Defaulting to full retrain due to diagnosis parse failure.",
        "estimated_impact_usd": 0.0,
        "full_report_markdown": f"# Drift Alert\n\nAlert ID: {alert.alert_id}\nSeverity: {alert.severity.value}\n\nDrifted features: {', '.join(alert.drifted_features)}\n\n**Diagnosis failed to parse. Manual review required.**",
    }
