"""Lightweight, explainable reranking without another model.

RRF gives a good candidate set, but it only uses ranks.  This reranker looks at
the actual evidence and asks four simple questions:

1. How similar is the chunk to the query?       (vector score)
2. How strong is the lexical overlap?            (normalised BM25)
3. How much of the query appears in the chunk?   (term coverage)
4. Does the heading itself match the query?      (heading coverage)

It is deliberately lexical: compliance evidence must stay explainable, and a
cross-encoder would add a large model dependency for a small corpus.
"""

from __future__ import annotations

from ..models import RetrievedChunk
from .bm25 import tokenize

# Weights were chosen on the 32-query retrieval set; they are constants rather
# than a grid search so the scoring stays easy to audit.
VECTOR_WEIGHT = 0.25
BM25_WEIGHT = 0.15
COVERAGE_WEIGHT = 0.35
HEADING_WEIGHT = 0.25


def rerank_results(query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if len(results) <= 1:
        return results
    query_terms = set(tokenize(query))
    if not query_terms:
        return results

    max_bm25 = max((item.bm25_score for item in results), default=0.0) or 1.0
    for item in results:
        text_terms = set(tokenize(item.chunk.embed_text))
        heading_terms = set(tokenize(" ".join(item.chunk.heading_path)))
        coverage = len(query_terms & text_terms) / len(query_terms)
        heading_coverage = len(query_terms & heading_terms) / len(query_terms)
        bm25_norm = min(1.0, item.bm25_score / max_bm25)
        item.rerank_score = round(
            VECTOR_WEIGHT * max(0.0, item.vector_score)
            + BM25_WEIGHT * bm25_norm
            + COVERAGE_WEIGHT * coverage
            + HEADING_WEIGHT * heading_coverage,
            6,
        )
    return sorted(results, key=lambda item: item.rerank_score, reverse=True)
