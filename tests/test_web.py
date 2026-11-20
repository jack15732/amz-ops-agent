from __future__ import annotations

import json

import httpx

from amz_agent.agent import RuleBasedAgent, ToolSession
from amz_agent.config import WebSearchSettings
from amz_agent.models import (
    AgentRequest,
    Decision,
    Site,
    WebResult,
    WebSearchResponse,
)
from amz_agent.rag.hybrid import HybridRetriever
from amz_agent.rag.vector_store import MemoryVectorStore
from amz_agent.web.client import TavilyClient


class FakeWeb:
    available = True

    def __init__(self):
        self.calls: list[str] = []

    def search(self, query, *, max_results=5, topic="general", days=None):
        self.calls.append(query)
        return WebSearchResponse(
            query=query,
            sufficient=True,
            results=[
                WebResult(
                    title="Amazon Seller Central — Product detail page rules",
                    url="https://sellercentral.amazon.com/help/hub/reference/example",
                    content="Best Seller claims require authorization and must be accurate.",
                    score=0.91,
                )
            ],
        )

    def close(self):
        pass


def test_tool_session_records_web_citations(service):
    session = ToolSession(
        AgentRequest(text="Cotton towel.", site=Site.US),
        service.retriever,
        service.engine,
        web=FakeWeb(),
    )
    assert "web_search" in {tool["function"]["name"] for tool in session.definitions()}
    result = session.execute("web_search", {"query": "Amazon best seller policy"})
    assert result["sufficient"] is True
    assert session.web_used and session.web_sufficient
    assert session.retrieval_rounds == 1
    assert session.citations[0].source_type == "web"
    assert session.citations[0].url.startswith("https://sellercentral.amazon.com")


def test_passed_listing_uses_web_when_kb_empty(service):
    retriever = HybridRetriever(
        MemoryVectorStore(), service.embedder, [], service.settings.retrieval, expander=None
    )
    agent = RuleBasedAgent(retriever, service.engine, web=FakeWeb())
    response = agent.run(
        AgentRequest(text="Plain cotton towel.", site=Site.US, question="可以上架吗？依据是什么？")
    )
    assert response.decision is Decision.PASSED
    assert response.web_used
    assert any(c.source_type == "web" for c in response.citations)
    assert response.guardrails.passed
    assert not response.handoff.required


def test_tavily_client_parses_results_without_exposing_key():
    settings = WebSearchSettings(enabled=True, api_key="tvly-test", include_domains=[])
    client = TavilyClient(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["api_key"] == "tvly-test"
        assert payload["query"] == "CE marking"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "CE marking",
                        "url": "https://europa.eu/ce-marking",
                        "content": "CE marking is a declaration.",
                        "score": 0.88,
                    }
                ]
            },
        )

    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    response = client.search("CE marking")
    assert response.sufficient
    assert response.results[0].url == "https://europa.eu/ce-marking"
    assert "tvly-test" not in response.model_dump_json()


def test_web_tool_is_hidden_without_client(service):
    session = ToolSession(
        AgentRequest(text="Cotton towel.", site=Site.US),
        service.retriever,
        service.engine,
        web=None,
    )
    assert "web_search" not in {tool["function"]["name"] for tool in session.definitions()}


def test_precheck_uses_web_when_kb_empty(service):
    from amz_agent.models import PrecheckRequest

    service.retriever = HybridRetriever(
        MemoryVectorStore(), service.embedder, [], service.settings.retrieval, expander=None
    )
    service.web = FakeWeb()
    response = service.precheck(
        PrecheckRequest(asin="B0TEST", site=Site.US, title="Plain cotton towel")
    )
    assert response.decision is Decision.PASSED
    assert any(c.source_type == "web" for c in response.citations)
    assert not response.handoff.required


def test_web_search_api(service, settings):
    from fastapi.testclient import TestClient
    from amz_agent.api.app import create_app

    service.web = FakeWeb()
    client = TestClient(create_app(settings, service))
    response = client.post(
        "/v1/web/search",
        json={"query": "Amazon best seller policy", "max_results": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sufficient"] is True
    assert body["results"][0]["url"].startswith("https://sellercentral.amazon.com")
