from __future__ import annotations

from fastapi.testclient import TestClient

from amz_agent.api.app import create_app


def test_health_and_compliance_endpoint(service, settings):
    app = create_app(settings, service)
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["chunks"] > 0

    response = client.post(
        "/v1/compliance/check",
        json={"text": "The best seller yoga mat.", "site": "US"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "blocked"
    assert body["matches"]

    trace = client.get(f"/v1/audit/{body['trace_id']}")
    assert trace.status_code == 200
    assert trace.json()["endpoint"] == "compliance"


def test_search_agent_and_precheck(service, settings):
    app = create_app(settings, service)
    client = TestClient(app)

    search = client.post(
        "/v1/knowledge/search", json={"query": "CE certified 为什么违规", "site": "EU"}
    )
    assert search.status_code == 200
    assert search.json()["results"]

    agent = client.post(
        "/v1/agent/query",
        json={
            "text": "Dermatologist tested formula.",
            "site": "US",
            "question": "需要人工复核吗？",
        },
    )
    assert agent.status_code == 200
    assert agent.json()["decision"] == "review"
    assert agent.json()["handoff"]["required"] is True

    precheck = client.post(
        "/v1/compliance/precheck",
        json={"asin": "B0TEST", "site": "US", "title": "Cotton towel", "bullets": ["Soft and absorbent"]},
    )
    assert precheck.status_code == 200
    assert precheck.json()["decision"] in {"passed", "review"}


def test_validation_errors(service, settings):
    app = create_app(settings, service)
    client = TestClient(app)
    assert client.post("/v1/compliance/check", json={"text": " ", "site": "US"}).status_code == 422
    assert client.get("/v1/audit/not-found").status_code == 404


def test_upload_ingest_keeps_existing_knowledge(service, settings):
    app = create_app(settings, service)
    client = TestClient(app)
    before = service.store.count()
    content = "---\ntitle: 上传测试\nsite: US\n---\n# 上传测试\n\n## 条款\n\n上传入库测试段落。\n"
    response = client.post(
        "/v1/knowledge/ingest",
        files={"files": ("upload_test.md", content.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] > before
    assert any(chunk.doc_id == "upload_test" for chunk in service.retriever.chunks)
