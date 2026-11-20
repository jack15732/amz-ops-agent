from __future__ import annotations

from amz_agent.models import Chunk, RetrievedChunk
from amz_agent.rag.rerank import rerank_results


def _chunk(chunk_id: str, heading: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        title="测试文档",
        source="doc.md",
        source_type="markdown",
        heading_path=["测试文档", heading],
        text=text,
    )


def test_rerank_promotes_query_terms_and_heading():
    query = "CE marking requirements"
    weak = RetrievedChunk(
        chunk=_chunk("a", "Shipping", "Delivery times and packaging details."),
        vector_score=0.55,
        bm25_score=0.1,
        score=0.03,
    )
    strong = RetrievedChunk(
        chunk=_chunk(
            "b",
            "CE marking requirements",
            "CE marking is a declaration of conformity for products sold in the EU.",
        ),
        vector_score=0.40,
        bm25_score=0.9,
        score=0.02,
    )
    ranked = rerank_results(query, [weak, strong])
    assert ranked[0].chunk.chunk_id == "b"
    assert ranked[0].rerank_score > ranked[1].rerank_score


def test_retriever_can_disable_rerank(service):
    query = "美国站能不能写 Best Seller"
    with_rerank, _ = service.retriever.search(query, top_k=5, use_rerank=True)
    without_rerank, _ = service.retriever.search(query, top_k=5, use_rerank=False)
    assert with_rerank and without_rerank
    assert all(item.rerank_score >= 0 for item in with_rerank)
