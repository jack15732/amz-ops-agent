from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Site(str, Enum):
    US = "US"
    EU = "EU"
    JP = "JP"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Decision(str, Enum):
    BLOCKED = "blocked"
    REVIEW = "review"
    PASSED = "passed"


class RuleAction(str, Enum):
    BLOCK = "block"
    REVIEW = "review"


class MatchType(str, Enum):
    TERM = "term"
    REGEX = "regex"


class LLMLabel(str, Enum):
    VIOLATION = "violation"
    BENIGN = "benign"
    UNCERTAIN = "uncertain"


class Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Rule(Base):
    id: str
    category: str
    severity: Severity
    action: RuleAction
    sites: list[Site] = Field(default_factory=lambda: list(Site))
    terms: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    allow_contexts: list[str] = Field(default_factory=list)
    boundary: Literal["auto", "none"] = "auto"
    case_sensitive: bool = False
    description: str = ""
    citation: str = ""
    suggestion: str = ""
    version: str = "v1"
    enabled: bool = True

    def applies_to(self, site: Site) -> bool:
        return self.enabled and site in self.sites


class MatchEvidence(Base):
    rule_id: str
    category: str
    severity: Severity
    action: RuleAction
    match_type: MatchType
    term: str
    start: int
    end: int
    snippet: str
    citation: str = ""
    suggestion: str = ""


class LLMReview(Base):
    label: LLMLabel
    confidence: float = 0.0
    reason: str = ""
    model: str = ""
    latency_ms: float = 0.0


class ComplianceResult(Base):
    decision: Decision
    risk_score: int = 0
    site: Site
    matches: list[MatchEvidence] = Field(default_factory=list)
    llm_review: LLMReview | None = None
    review_required: bool = False
    summary: str = ""
    rule_version: str = ""
    latency_ms: float = 0.0
    trace_id: str = ""

    @property
    def blocked(self) -> bool:
        return self.decision is Decision.BLOCKED


class Chunk(Base):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    source_type: str
    heading_path: list[str] = Field(default_factory=list)
    text: str
    embed_text: str = ""
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.embed_text:
            prefix = " > ".join(self.heading_path)
            self.embed_text = f"{self.title} > {prefix}\n{self.text}" if prefix else f"{self.title}\n{self.text}"

    @staticmethod
    def make_id(doc_id: str, heading_path: list[str], index: int, text: str) -> str:
        raw = f"{doc_id}|{'/'.join(heading_path)}|{index}|{text[:80]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class Citation(Base):
    chunk_id: str
    doc_id: str
    title: str
    heading_path: list[str] = Field(default_factory=list)
    source: str = ""
    snippet: str = ""
    score: float = 0.0
    source_type: Literal["knowledge", "web"] = "knowledge"
    url: str = ""
    published_date: str = ""
    official: bool = False


class RetrievedChunk(Base):
    chunk: Chunk
    score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float = 0.0
    matched_terms: list[str] = Field(default_factory=list)

    def citation(self) -> Citation:
        snippet = self.chunk.text.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        return Citation(
            chunk_id=self.chunk.chunk_id,
            doc_id=self.chunk.doc_id,
            title=self.chunk.title,
            heading_path=self.chunk.heading_path,
            source=self.chunk.source,
            snippet=snippet,
            score=round(self.score, 6),
        )


class SearchRequest(Base):
    query: str
    site: Site | None = None
    top_k: int = 6
    use_expansion: bool = True


class SearchResponse(Base):
    query: str
    results: list[RetrievedChunk] = Field(default_factory=list)
    rewritten_query: str = ""
    sufficient: bool = False
    latency_ms: float = 0.0
    trace_id: str = ""


class WebResult(Base):
    title: str
    url: str
    content: str = ""
    score: float = 0.0
    published_date: str = ""
    official: bool = False


class WebSearchRequest(Base):
    query: str
    max_results: int = 5
    topic: Literal["general", "news"] = "general"
    days: int | None = None


class WebSearchResponse(Base):
    query: str
    results: list[WebResult] = Field(default_factory=list)
    sufficient: bool = False
    provider: str = "tavily"
    latency_ms: float = 0.0
    error: str = ""


class ComplianceRequest(Base):
    text: str
    site: Site
    content_type: str = "listing"
    trace_id: str | None = None


class PrecheckRequest(BaseModel):
    """One-shot listing pre-review payload used by the listing pipeline."""

    model_config = ConfigDict(extra="ignore")
    asin: str | None = None
    site: Site = Site.US
    title: str = ""
    bullets: list[str] = Field(default_factory=list)
    description: str = ""
    a_plus: str = ""
    trace_id: str | None = None

    def full_text(self) -> str:
        parts = [self.title, *self.bullets, self.description, self.a_plus]
        return "\n".join(p for p in parts if p.strip())


class AgentRequest(Base):
    text: str
    site: Site = Site.US
    question: str = "请对这段 Listing 文案做上架前合规预审，并给出修改建议。"
    content_type: str = "listing"
    session_id: str = "default"
    trace_id: str | None = None


class ToolCallRecord(Base):
    seq: int
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    result_count: int = 0


class GuardrailReport(Base):
    passed: bool = True
    compliance_checked: bool = False
    exact_text_checked: bool = False
    citations_valid: bool = False
    evidence_sufficient: bool = False
    overrides: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class Handoff(Base):
    required: bool = False
    reason: str = ""
    queue: str = "compliance_review"
    priority: str = "normal"
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(Base):
    answer: str
    decision: Decision
    risk_score: int = 0
    site: Site
    matches: list[MatchEvidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    handoff: Handoff = Field(default_factory=Handoff)
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    agent_mode: str = "rule_based"
    web_used: bool = False
    retrieval_rounds: int = 0
    rule_version: str = ""
    trace_id: str = ""
    latency_ms: float = 0.0
    llm_review: LLMReview | None = None


class PrecheckResponse(Base):
    asin: str | None = None
    site: Site
    decision: Decision
    risk_score: int = 0
    matches: list[MatchEvidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    handoff: Handoff = Field(default_factory=Handoff)
    summary: str = ""
    trace_id: str = ""
    latency_ms: float = 0.0


class AuditTrace(Base):
    trace_id: str
    endpoint: str
    site: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    status: str = "ok"
    created_at: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class HealthResponse(Base):
    status: str = "ok"
    version: str
    embedding_provider: str
    embedding_model: str
    llm_available: bool
    llm_model: str = ""
    web_search_available: bool = False
    qdrant_mode: str
    chunks: int
    rules: int
