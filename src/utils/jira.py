"""
Jira REST API integration for DriftSentinel.

Used by the canary promoter to file rollback tickets automatically.
"""

from __future__ import annotations

import asyncio

import httpx

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def create_jira_ticket(
    project: str,
    summary: str,
    description: str,
    issue_type: str = "Bug",
    labels: list[str] | None = None,
    priority: str = "High",
) -> str | None:
    """
    Create a Jira issue via the REST API v3.

    Returns the created issue key (e.g. "DS-42") or None on failure.
    """
    cfg = settings.jira
    if not cfg.base_url or not cfg.api_token:
        logger.warning("jira_not_configured_skipping_ticket")
        return None

    url = f"{cfg.base_url.rstrip('/')}/rest/api/3/issue"

    payload = {
        "fields": {
            "project": {"key": project},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}],
                    }
                ],
            },
            "issuetype": {"name": issue_type},
            "priority": {"name": priority},
            "labels": labels or [],
        }
    }

    auth = (cfg.email, cfg.api_token)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, auth=auth)
            resp.raise_for_status()
            issue_key = resp.json().get("key")
            logger.info("jira_ticket_created", key=issue_key, summary=summary[:80])
            return issue_key
    except httpx.HTTPStatusError as exc:
        logger.error(
            "jira_http_error",
            status=exc.response.status_code,
            body=exc.response.text[:200],
        )
        return None
    except Exception as exc:
        logger.error("jira_error", error=str(exc))
        return None
