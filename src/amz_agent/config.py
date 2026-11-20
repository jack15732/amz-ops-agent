"""Configuration loading.

Priority: environment variables > YAML config > defaults.
The project intentionally has no python-dotenv dependency; a small parser keeps
local setup to a single `cp .env.example .env`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

def _detect_project_root() -> Path:
    """Work from a source checkout, an editable install, or a Docker WORKDIR."""
    env_root = os.environ.get("AMZ_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "settings.yaml").exists():
            return parent
    cwd = Path.cwd()
    if (cwd / "config" / "settings.yaml").exists():
        return cwd
    return here.parents[2]


PROJECT_ROOT = _detect_project_root()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class AppSettings(_Base):
    name: str = "AMZ Ops Agent"
    version: str = "0.1.0"
    data_dir: str = "data/runtime"
    log_level: str = "INFO"
    auto_ingest: bool = True


class EmbeddingSettings(_Base):
    provider: str = "siliconflow"
    model: str = "BAAI/bge-m3"
    fallback_model: str = "moka-ai/m3e-base"
    dimension: int | None = None
    batch_size: int = 16
    timeout_s: float = 60.0
    cache_size: int = 2048
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"


class LLMSettings(_Base):
    provider: str = "siliconflow"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 1500
    timeout_s: float = 60.0
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"


class RetrievalSettings(_Base):
    top_k: int = 6
    candidate_k: int = 30
    rrf_k: int = 60
    vector_weight: float = 1.2
    bm25_weight: float = 1.0
    min_score: float = 0.008
    min_vector_score: float = 0.30
    min_bm25_score: float = 0.80
    max_chunk_chars: int = 760
    chunk_overlap: int = 120
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    query_expansion: bool = True


class ComplianceSettings(_Base):
    rule_dir: str = "config/compliance"
    review_on_unknown: bool = True
    llm_review_threshold: float = 0.80


class AgentSettings(_Base):
    max_steps: int = 6
    min_citations: int = 1
    require_compliance_check: bool = True
    allow_human_handoff: bool = True


class AuditSettings(_Base):
    sqlite_path: str = "data/runtime/audit.sqlite3"
    keep_days: int = 180


class QdrantSettings(_Base):
    mode: str = "local"  # local | server | memory
    url: str = "http://localhost:6333"
    path: str = ""
    api_key: str = ""
    collection: str = "amz_knowledge"


class Settings(_Base):
    app: AppSettings = Field(default_factory=AppSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    compliance: ComplianceSettings = Field(default_factory=ComplianceSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def data_path(self) -> Path:
        return self.resolve_path(self.app.data_dir)

    @property
    def rule_path(self) -> Path:
        return self.resolve_path(self.compliance.rule_dir)

    @property
    def audit_path(self) -> Path:
        return self.resolve_path(self.audit.sqlite_path)

    @property
    def qdrant_path(self) -> Path:
        return self.data_path / "qdrant"


def _apply_env(settings: Settings) -> Settings:
    env = os.environ
    key = env.get("SILICONFLOW_API_KEY", "")

    settings.embedding.provider = env.get("EMBEDDING_PROVIDER", settings.embedding.provider)
    settings.embedding.model = env.get("EMBEDDING_MODEL", settings.embedding.model)
    settings.embedding.base_url = env.get(
        "SILICONFLOW_BASE_URL", settings.embedding.base_url
    )
    settings.embedding.api_key = env.get("SILICONFLOW_API_KEY", settings.embedding.api_key) or key

    settings.llm.provider = env.get("LLM_PROVIDER", settings.llm.provider)
    settings.llm.model = env.get("LLM_MODEL", settings.llm.model)
    settings.llm.base_url = env.get("SILICONFLOW_BASE_URL", settings.llm.base_url)
    settings.llm.api_key = env.get("SILICONFLOW_API_KEY", settings.llm.api_key) or key

    settings.app.data_dir = env.get("AMZ_DATA_DIR", settings.app.data_dir)
    settings.app.log_level = env.get("AMZ_LOG_LEVEL", settings.app.log_level)
    if "AMZ_AUTO_INGEST" in env:
        settings.app.auto_ingest = env["AMZ_AUTO_INGEST"].lower() in {"1", "true", "yes"}
    settings.qdrant.mode = env.get("AMZ_QDRANT_MODE", settings.qdrant.mode)
    settings.qdrant.url = env.get("QDRANT_URL", settings.qdrant.url)
    settings.qdrant.api_key = env.get("QDRANT_API_KEY", settings.qdrant.api_key)

    # `none` is a supported explicit offline mode.
    if settings.llm.provider == "none":
        settings.llm.api_key = ""
    return settings


def load_settings(path: str | Path | None = None) -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")
    config_path = Path(path or os.environ.get("AMZ_CONFIG", PROJECT_ROOT / "config/settings.yaml"))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    settings = _apply_env(Settings.model_validate(raw))
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.audit_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
