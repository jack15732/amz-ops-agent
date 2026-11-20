from __future__ import annotations

from amz_agent.models import SearchRequest, Site
from amz_agent.rag.chunker import Chunker, Section
from amz_agent.rag.hybrid import QueryExpander


def _section(text: str, heading: str = "测试章节") -> Section:
    return Section(
        doc_id="doc",
        title="测试文档",
        source="doc.md",
        source_type="markdown",
        heading_path=["测试文档", heading],
        text=text,
    )


def test_chunker_keeps_heading_path_and_overlap():
    chunker = Chunker(max_chars=200, overlap=40)
    text = "第一句。" * 60
    chunks = chunker.split(_section(text))
    assert len(chunks) >= 2
    assert all(c.heading_path == ["测试文档", "测试章节"] for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_chunk_id_is_stable():
    chunker = Chunker(max_chars=200, overlap=40)
    first = chunker.split(_section("同样的文本。" * 20))
    second = chunker.split(_section("同样的文本。" * 20))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_bm25_and_hybrid_retrieval(service):
    response = service.search(
        SearchRequest(query="美国站能不能写 Best Seller", site=Site.US, top_k=3)
    )
    assert response.results
    assert response.sufficient
    assert any("Best Seller" in " ".join(r.chunk.heading_path) for r in response.results)


def test_no_answer_guard(service):
    response = service.search(
        SearchRequest(query="火星站需要什么星际认证？", site=Site.US, top_k=3)
    )
    assert response.sufficient is False


def test_query_expansion():
    expander = QueryExpander("config/lexicon.yaml")
    expanded, additions = expander.expand("BSR 怎么提升？")
    assert "Best Sellers Rank" in expanded
    assert additions
