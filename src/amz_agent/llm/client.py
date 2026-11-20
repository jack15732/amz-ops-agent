from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from ..config import LLMSettings
from ..models import LLMLabel, LLMReview, MatchEvidence, Site
from .prompts import COMPLIANCE_REVIEW_PROMPT, review_user_prompt

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Minimal OpenAI-compatible chat client used for tool calling and review."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._client = httpx.Client(timeout=settings.timeout_s)

    @property
    def available(self) -> bool:
        return bool(self.settings.api_key) and self.settings.provider != "none"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            raise LLMError("LLM API key is not configured")
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(url, headers=headers, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise LLMError(f"upstream {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.8 * (2**attempt))
        raise LLMError(f"LLM request failed: {last_error}")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._post(payload)
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected completion shape: {data}") from exc

    def chat_json(self, system: str, user: str) -> dict[str, Any] | None:
        message = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=400,
        )
        content = message.get("content") or ""
        return _extract_json(content)

    def review_compliance(
        self, text: str, site: Site, matches: list[MatchEvidence]
    ) -> LLMReview | None:
        if not self.available:
            return None
        started = time.perf_counter()
        hits = [
            {
                "rule_id": m.rule_id,
                "severity": m.severity.value,
                "term": m.term,
                "snippet": m.snippet,
            }
            for m in matches
        ]
        data = self.chat_json(
            COMPLIANCE_REVIEW_PROMPT, review_user_prompt(site.value, text, hits)
        )
        if not data:
            return None
        try:
            label = LLMLabel(str(data.get("label", "uncertain")))
        except ValueError:
            label = LLMLabel.UNCERTAIN
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0))))
        return LLMReview(
            label=label,
            confidence=confidence,
            reason=str(data.get("reason", ""))[:200],
            model=self.settings.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def close(self) -> None:
        self._client.close()


def _extract_json(content: str) -> dict[str, Any] | None:
    content = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    if fenced:
        content = fenced.group(1)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON content: %s", content[:200])
        return None
