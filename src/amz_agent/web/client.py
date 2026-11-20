"""Tavily web search adapter.

The API key is read from the environment and never written to logs or audit
payloads.  Web pages are untrusted data: callers must treat the returned text
as evidence only, not as instructions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ..config import WebSearchSettings
from ..models import WebResult, WebSearchResponse
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class TavilyClient:
    def __init__(self, settings: WebSearchSettings):
        self.settings = settings
        self._client = httpx.Client(timeout=settings.timeout_s)
        self._available = bool(settings.enabled and settings.api_key)

    @property
    def available(self) -> bool:
        return self._available

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        topic: str = "general",
        days: int | None = None,
    ) -> WebSearchResponse:
        started = time.perf_counter()
        if not self.available:
            return WebSearchResponse(
                query=query,
                sufficient=False,
                error="Tavily is not configured (set TAVILY_API_KEY)",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        if not query.strip():
            return WebSearchResponse(query=query, sufficient=False, error="empty query")

        limit = min(10, max(1, max_results or self.settings.max_results))
        payload: dict[str, Any] = {
            "api_key": self.settings.api_key,
            "query": query,
            "search_depth": self.settings.search_depth,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": limit,
            "topic": topic if topic in {"general", "news"} else "general",
        }
        if days and payload["topic"] == "news":
            payload["days"] = int(days)
        if self.settings.include_domains:
            payload["include_domains"] = self.settings.include_domains

        data = self._post(payload)
        results: list[WebResult] = []
        for row in data.get("results", []) or []:
            url = str(row.get("url", "")).strip()
            if not url:
                continue
            official = _is_official(url, self.settings.priority_domains)
            results.append(
                WebResult(
                    title=str(row.get("title", url))[:300],
                    url=url,
                    content=str(row.get("content", ""))[:2000],
                    score=float(row.get("score", 0.0) or 0.0),
                    published_date=str(row.get("published_date", "") or ""),
                    official=official,
                )
            )
        # Official regulator/vendor domains first; score within each group.
        results.sort(key=lambda item: (item.official, item.score), reverse=True)
        sufficient = bool(results) and max(item.score for item in results) >= 0.20
        return WebSearchResponse(
            query=query,
            results=results[:limit],
            sufficient=sufficient,
            provider="tavily",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}/search"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(url, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"Tavily HTTP {response.status_code}")
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("unexpected Tavily response")
                return data
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.8 * (2**attempt))
        logger.warning("Tavily search failed: %s", last_error)
        return {"results": []}

    def close(self) -> None:
        self._client.close()


def _is_official(url: str, domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def build_web_client(settings: WebSearchSettings) -> TavilyClient:
    return TavilyClient(settings)
