"""
RAG Pipeline — Past Incident Retrieval via Pinecone

Incident embeddings are generated from a structured text representation of
each DiagnosisReport (model_id, drifted features, severity, hypothesis, strategy).

On every new alert:
  1. Generate query embedding from the alert's statistical summary
  2. Retrieve top-k similar past incidents from Pinecone
  3. Return them as context for the LLM agent

After each diagnosis:
  1. Generate embedding of the full DiagnosisReport
  2. Upsert into Pinecone with metadata for filtering

As the organization accumulates incident history, the agent gets smarter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec

from src.ingestion.schema import DriftAlert, DiagnosisReport
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
INDEX_NAME = "driftsentinel-incidents"


@dataclass
class IncidentRAG:
    """
    Manages the Pinecone incident vector store.
    Provides semantic retrieval of past drift incidents.
    """

    _pc: Pinecone = field(init=False)
    _index: Any = field(init=False)
    _openai: AsyncOpenAI = field(init=False)

    def __post_init__(self) -> None:
        self._pc = Pinecone(api_key=settings.pinecone.api_key)
        self._ensure_index()
        self._index = self._pc.Index(INDEX_NAME)
        self._openai = AsyncOpenAI(api_key=settings.llm.api_key)
        logger.info("rag_initialized", index=INDEX_NAME)

    def _ensure_index(self) -> None:
        existing = [idx.name for idx in self._pc.list_indexes()]
        if INDEX_NAME not in existing:
            self._pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone.cloud,
                    region=settings.pinecone.region,
                ),
            )
            logger.info("pinecone_index_created", name=INDEX_NAME)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve_similar(
        self,
        alert: DriftAlert,
        top_k: int = 3,
        model_id_filter: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k most semantically similar past incidents.
        Optionally filter by same model_id for higher relevance.
        """
        query_text = _alert_to_text(alert)
        embedding = await self._embed(query_text)

        filter_dict: dict | None = None
        if model_id_filter:
            filter_dict = {"model_id": {"$eq": alert.model_id}}

        results = self._index.query(
            vector=embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
        )

        # If no model-specific incidents, fall back to global search
        if not results.matches and model_id_filter:
            results = self._index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
            )

        incidents = []
        for match in results.matches:
            incidents.append({
                "incident_id": match.id,
                "similarity_score": float(match.score),
                "model_id": match.metadata.get("model_id"),
                "occurred_at": match.metadata.get("occurred_at"),
                "severity": match.metadata.get("severity"),
                "drifted_features": match.metadata.get("drifted_features", []),
                "top_hypothesis": match.metadata.get("top_hypothesis"),
                "resolved_strategy": match.metadata.get("recommended_strategy"),
                "summary": match.metadata.get("summary", ""),
            })

        logger.info(
            "rag_retrieved",
            alert_id=alert.alert_id,
            incidents_found=len(incidents),
        )
        return incidents

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_incident(self, report: DiagnosisReport) -> str:
        """Embed and upsert a resolved incident into the vector store."""
        text = _report_to_text(report)
        embedding = await self._embed(text)

        vector_id = _incident_id(report)
        metadata = {
            "model_id": report.model_id,
            "alert_id": report.alert_id,
            "diagnosis_id": report.diagnosis_id,
            "occurred_at": report.generated_at.isoformat(),
            "severity": report.hypotheses[0].get("confidence", 0) if report.hypotheses else 0,
            "drifted_features": [],  # populated from alert context
            "top_hypothesis": report.top_hypothesis,
            "recommended_strategy": report.recommended_strategy.value,
            "strategy_rationale": report.strategy_rationale[:500],
            "summary": report.top_hypothesis[:200],
            "top_confidence": report.top_hypothesis_confidence,
        }

        self._index.upsert(vectors=[(vector_id, embedding, metadata)])
        logger.info("incident_indexed", vector_id=vector_id, model_id=report.model_id)
        return vector_id

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        response = await self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000],  # Token limit
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def index_stats(self) -> dict:
        stats = self._index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": stats.namespaces,
        }


# ---------------------------------------------------------------------------
# Text serializers for embedding
# ---------------------------------------------------------------------------

def _alert_to_text(alert: DriftAlert) -> str:
    """
    Convert a DriftAlert to a text representation for embedding.
    Structured to match the format of indexed past incidents.
    """
    test_lines = []
    for r in alert.test_results:
        if r.drifted:
            test_lines.append(
                f"  - {r.test_name} on {r.feature_name}: statistic={r.statistic:.4f}"
            )

    return f"""
Model: {alert.model_id}
Severity: {alert.severity.value}
Segment: {json.dumps(alert.segment)}
Drifted features: {', '.join(alert.drifted_features)}
Statistical tests fired:
{chr(10).join(test_lines)}
Window: {alert.window_start.isoformat()} to {alert.window_end.isoformat()}
""".strip()


def _report_to_text(report: DiagnosisReport) -> str:
    return f"""
Model: {report.model_id}
Root cause: {report.top_hypothesis}
Strategy applied: {report.recommended_strategy.value}
Rationale: {report.strategy_rationale}
Affected segments: {json.dumps(report.affected_segments)}
Impact estimate: ${report.estimated_impact_usd:.2f}
""".strip()


def _incident_id(report: DiagnosisReport) -> str:
    """Deterministic ID: same alert always maps to same vector."""
    return hashlib.sha256(
        f"{report.alert_id}:{report.diagnosis_id}".encode()
    ).hexdigest()[:40]
