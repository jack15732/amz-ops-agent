from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import yaml

from ..config import RetrievalSettings
from ..models import Chunk, RetrievedChunk, Site
from .bm25 import BM25Index, tokenize
from .embeddings import Embedder
from .rerank import rerank_results
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class QueryExpander:
    """Bidirectional alias expansion for cross-border abbreviations.

    Entry format:
      - canonical: BSR
        aliases: ["Best Sellers Rank", "畅销榜", "销售排名"]
    """

    def __init__(self, path: str | Path | None = None):
        self.groups: list[list[str]] = []
        self.lookup: list[re.Pattern[str]] = []
        if path and Path(path).exists():
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            for entry in raw.get("terms", []):
                group = [entry.get("canonical", "")] + list(entry.get("aliases", []))
                group = [term for term in group if term]
                if len(group) >= 2:
                    self.groups.append(group)
            for group in self.groups:
                alternatives = []
                for term in group:
                    if term.isascii():
                        alternatives.append(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
                    else:
                        alternatives.append(re.escape(term))
                self.lookup.append(re.compile("|".join(alternatives), re.IGNORECASE))

    def expand(self, query: str) -> tuple[str, list[str]]:
        additions: list[str] = []
        for group, pattern in zip(self.groups, self.lookup):
            if pattern.search(query):
                additions.extend(term for term in group if term.lower() not in query.lower())
        if not additions:
            return query, []
        # Keep the original query first; aliases are only a recall booster.
        return f"{query} {' '.join(dict.fromkeys(additions))}", list(dict.fromkeys(additions))


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunks: list[Chunk],
        settings: RetrievalSettings,
        *,
        expander: QueryExpander | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.chunks = chunks
        self.settings = settings
        self.expander = expander
        self.bm25 = BM25Index(chunks, k1=settings.bm25_k1, b=settings.bm25_b)

    def reload(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Index(chunks, k1=self.settings.bm25_k1, b=self.settings.bm25_b)

    def search(
        self,
        query: str,
        *,
        site: Site | None = None,
        top_k: int | None = None,
        use_expansion: bool | None = None,
        use_rerank: bool | None = None,
    ) -> tuple[list[RetrievedChunk], str]:
        top_k = top_k or self.settings.top_k
        use_expansion = self.settings.query_expansion if use_expansion is None else use_expansion
        rewritten, _ = (
            self.expander.expand(query)
            if (use_expansion and self.expander is not None)
            else (query, [])
        )
        site_value = site.value if site else None

        query_vector = self.embedder.embed([rewritten])[0]
        vector_hits = self.store.search(
            query_vector, self.settings.candidate_k, site=site_value
        )
        bm25_hits = self.bm25.search(
            tokenize(rewritten), self.settings.candidate_k, site=site_value
        )

        fused: dict[str, RetrievedChunk] = {}
        for rank, (chunk, score) in enumerate(vector_hits, start=1):
            # A vector hit below the relevance floor only adds noise to RRF.
            if score < self.settings.min_vector_score:
                continue
            fused[chunk.chunk_id] = RetrievedChunk(
                chunk=chunk,
                vector_score=score,
                vector_rank=rank,
                score=self.settings.vector_weight / (self.settings.rrf_k + rank),
            )
        for rank, (chunk, score, matched) in enumerate(bm25_hits, start=1):
            item = fused.get(chunk.chunk_id)
            contribution = self.settings.bm25_weight / (self.settings.rrf_k + rank)
            if item is None:
                fused[chunk.chunk_id] = RetrievedChunk(
                    chunk=chunk,
                    bm25_score=score,
                    bm25_rank=rank,
                    matched_terms=matched,
                    score=contribution,
                )
            else:
                item.bm25_score = score
                item.bm25_rank = rank
                item.matched_terms = matched
                item.score += contribution

        ranked = sorted(fused.values(), key=lambda item: item.score, reverse=True)
        should_rerank = self.settings.rerank_enabled if use_rerank is None else use_rerank
        if should_rerank and len(ranked) > 1:
            candidates = ranked[: self.settings.rerank_candidate_k]
            ranked = rerank_results(query, candidates)
        return ranked[:top_k], rewritten

    def is_sufficient(self, results: list[RetrievedChunk], query: str = "") -> bool:
        if not results:
            return False
        top = results[0]
        if top.score < self.settings.min_score:
            return False
        has_vector = top.vector_score >= self.settings.min_vector_score
        has_bm25 = top.bm25_score >= self.settings.min_bm25_score
        if not (has_vector or has_bm25):
            return False
        if query:
            # Reject queries whose key entities never occur in the corpus
            # ("火星站", "2099 量子条款").  Generic words such as 合规/认证 are
            # not enough to ground an answer.  A very strong BM25 hit still wins.
            tokens = list(dict.fromkeys(tokenize(query)))
            if tokens:
                vocab = self.bm25.idf
                absent = sum(len(t) for t in tokens if t not in vocab)
                total = sum(len(t) for t in tokens)
                oov_ratio = absent / total if total else 0.0
                if oov_ratio > 0.45 and top.bm25_score < 25:
                    return False
        return True

    def search_response(
        self,
        query: str,
        *,
        site: Site | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
    ):
        from ..models import SearchResponse

        started = time.perf_counter()
        results, rewritten = self.search(
            query, site=site, top_k=top_k, use_rerank=use_rerank
        )
        return SearchResponse(
            query=query,
            results=results,
            rewritten_query=rewritten,
            sufficient=self.is_sufficient(results, query),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
