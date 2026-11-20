from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentService
from .agent.tools import text_hash
from .audit import AuditStore
from .compliance import ComplianceEngine, load_rules
from .config import PROJECT_ROOT, Settings, load_settings
from .llm import LLMClient
from .models import (
    AgentRequest,
    AgentResponse,
    AuditTrace,
    Citation,
    ComplianceRequest,
    ComplianceResult,
    Decision,
    Handoff,
    HealthResponse,
    PrecheckRequest,
    PrecheckResponse,
    SearchRequest,
    SearchResponse,
    Site,
)
from .rag import Chunker, HybridRetriever, QueryExpander, build_embedder, ingest_paths
from .rag.ingest import load_catalog
from .rag.vector_store import build_vector_store

logger = logging.getLogger(__name__)


class AppService:
    """Wires the rule engine, hybrid retriever, agent and audit trail."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.data_path = self.settings.data_path
        self.catalog_path = self.data_path / "chunks.jsonl"
        self.knowledge_path = PROJECT_ROOT / "data" / "knowledge"

        self.ruleset = load_rules(self.settings.rule_path)
        self.llm = LLMClient(self.settings.llm)
        self.engine = ComplianceEngine(
            self.ruleset,
            reviewer=self.llm if self.llm.available else None,
            llm_review_threshold=self.settings.compliance.llm_review_threshold,
        )
        self.embedder = build_embedder(self.settings.embedding)
        self.store = build_vector_store(self.settings.qdrant, self.data_path)
        self.audit = AuditStore(self.settings.audit_path)

        chunks = load_catalog(self.catalog_path)
        if self.settings.app.auto_ingest and (
            not chunks or self.store.count() == 0
        ):
            logger.info("knowledge index empty, ingesting %s", self.knowledge_path)
            stats = self.ingest([self.knowledge_path], reset=True)
            logger.info("auto ingest: %s", stats)
            chunks = load_catalog(self.catalog_path)
        elif chunks and self.store.count() != len(chunks):
            # Recover a missing/damaged catalog from the vector payloads.
            stored = self.store.all_chunks()
            if stored:
                chunks = stored
                self._write_catalog(chunks)

        self.retriever = HybridRetriever(
            self.store,
            self.embedder,
            chunks,
            self.settings.retrieval,
            expander=QueryExpander(PROJECT_ROOT / "config" / "lexicon.yaml"),
        )
        self.agent = AgentService(
            self.retriever,
            self.engine,
            llm=self.llm,
            settings=self.settings.agent,
        )

    # ------------------------------------------------------------------ ingest
    def ingest(self, paths: list[str | Path], *, reset: bool = False) -> dict:
        stats = ingest_paths(
            paths,
            self.embedder,
            self.store,
            chunker=Chunker(
                max_chars=self.settings.retrieval.max_chunk_chars,
                overlap=self.settings.retrieval.chunk_overlap,
            ),
            catalog_path=self.catalog_path,
            reset=reset,
        )
        chunks = load_catalog(self.catalog_path)
        if getattr(self, "retriever", None) is not None:
            self.retriever.reload(chunks)
        return stats

    def _write_catalog(self, chunks) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with self.catalog_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")

    # --------------------------------------------------------------- endpoints
    def check(self, request: ComplianceRequest) -> ComplianceResult:
        trace_id = request.trace_id or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        result = self.engine.check(request.text, request.site, trace_id=trace_id)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.audit.record_trace(
            AuditTrace(
                trace_id=trace_id,
                endpoint="compliance",
                site=request.site.value,
                input=request.model_dump(mode="json"),
                output=result.model_dump(mode="json"),
                latency_ms=result.latency_ms,
                created_at=_now(),
            )
        )
        self.audit.record_compliance(trace_id, result, text_hash(request.text))
        return result

    def search(self, request: SearchRequest) -> SearchResponse:
        trace_id = uuid.uuid4().hex[:16]
        started = time.perf_counter()
        response = self.retriever.search_response(
            request.query,
            site=request.site,
            top_k=request.top_k,
        )
        response.trace_id = trace_id
        response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.audit.record_trace(
            AuditTrace(
                trace_id=trace_id,
                endpoint="search",
                site=request.site.value if request.site else "",
                input=request.model_dump(mode="json"),
                output=response.model_dump(mode="json"),
                latency_ms=response.latency_ms,
                created_at=_now(),
            )
        )
        return response

    def run_agent(self, request: AgentRequest) -> AgentResponse:
        response = self.agent.run(request)
        trace_id = response.trace_id
        self.audit.record_agent(trace_id, request, response)
        compliance = ComplianceResult(
            decision=response.decision,
            risk_score=response.risk_score,
            site=response.site,
            matches=response.matches,
            llm_review=response.llm_review,
            review_required=response.decision is Decision.REVIEW,
            rule_version=response.rule_version,
            trace_id=trace_id,
        )
        self.audit.record_compliance(trace_id, compliance, text_hash(request.text))
        return response

    def precheck(self, request: PrecheckRequest) -> PrecheckResponse:
        trace_id = request.trace_id or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        text = request.full_text()
        result = self.engine.check(text, request.site, trace_id=trace_id)
        search = self.retriever.search_response(
            f"{request.site.value} listing 合规 预审 要求",
            site=request.site,
            top_k=3,
        )
        citations: list[Citation] = [item.citation() for item in search.results]
        decision = result.decision
        handoff = Handoff()
        if decision is Decision.REVIEW:
            handoff = Handoff(
                required=True,
                reason=result.summary or "规则命中待复核内容",
                priority="high",
                payload={"asin": request.asin, "trace_id": trace_id},
            )
        elif decision is Decision.PASSED and not search.sufficient:
            decision = Decision.REVIEW
            handoff = Handoff(
                required=True,
                reason="规则未命中，但知识库无可靠检索依据",
                priority="normal",
                payload={"asin": request.asin, "trace_id": trace_id},
            )
        latency = round((time.perf_counter() - started) * 1000, 2)
        response = PrecheckResponse(
            asin=request.asin,
            site=request.site,
            decision=decision,
            risk_score=result.risk_score if decision is not Decision.REVIEW else max(result.risk_score, 20),
            matches=result.matches,
            citations=citations,
            handoff=handoff,
            summary=result.summary,
            trace_id=trace_id,
            latency_ms=latency,
        )
        self.audit.record_trace(
            AuditTrace(
                trace_id=trace_id,
                endpoint="precheck",
                site=request.site.value,
                input=request.model_dump(mode="json"),
                output=response.model_dump(mode="json"),
                latency_ms=latency,
                created_at=_now(),
            )
        )
        self.audit.record_compliance(trace_id, result, text_hash(text))
        return response

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=self.settings.app.version,
            embedding_provider=self.embedder.name,
            embedding_model=self.embedder.model,
            llm_available=self.llm.available,
            qdrant_mode=self.settings.qdrant.mode,
            chunks=self.store.count(),
            rules=self.engine.rule_count,
        )

    def close(self) -> None:
        self.llm.close()
        self.audit.close()
        close = getattr(self.embedder, "close", None)
        if close:
            close()


def build_service(settings: Settings | None = None) -> AppService:
    return AppService(settings)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
