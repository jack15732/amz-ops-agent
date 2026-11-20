from __future__ import annotations

from pathlib import Path

import pytest

from amz_agent.config import Settings, load_settings
from amz_agent.service import AppService


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    settings = load_settings()
    settings.app.data_dir = str(tmp_path / "data")
    settings.app.auto_ingest = True
    settings.qdrant.mode = "memory"
    settings.embedding.provider = "hash"
    settings.llm.provider = "none"
    settings.llm.api_key = ""
    settings.web_search.enabled = False
    settings.web_search.api_key = ""
    settings.retrieval.query_expansion = True
    return settings


@pytest.fixture()
def service(settings: Settings):
    svc = AppService(settings)
    yield svc
    svc.close()
