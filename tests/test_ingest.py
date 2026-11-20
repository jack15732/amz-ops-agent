from __future__ import annotations

from pathlib import Path

from amz_agent.config import Settings
from amz_agent.models import SearchRequest, Site
from amz_agent.rag.ingest import read_sections
from amz_agent.service import AppService


def test_docx_and_xlsx_sections(tmp_path: Path):
    from docx import Document
    from openpyxl import Workbook

    docx_path = tmp_path / "sop.docx"
    doc = Document()
    doc.add_heading("预审流程", level=1)
    doc.add_heading("第一步", level=2)
    doc.add_paragraph("核对标题与五点描述。")
    doc.save(docx_path)

    xlsx_path = tmp_path / "rules.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "US"
    ws.append(["规则ID", "说明"])
    ws.append(["US-ABS-001", "绝对化用语"])
    wb.save(xlsx_path)

    docx_sections = read_sections(docx_path)
    assert docx_sections[0].heading_path[-1] == "第一步"
    assert "核对标题" in docx_sections[0].text

    xlsx_sections = read_sections(xlsx_path)
    assert xlsx_sections[0].sheet == "US"
    assert xlsx_sections[0].row_start == 1
    assert "US-ABS-001" in xlsx_sections[0].text


def test_qdrant_local_mode(tmp_path: Path):
    from amz_agent.config import load_settings
    from amz_agent.models import SearchRequest, Site

    settings = load_settings()
    settings.app.data_dir = str(tmp_path / "qdrant_data")
    settings.qdrant.mode = "local"
    settings.embedding.provider = "hash"
    settings.llm.provider = "none"
    settings.app.auto_ingest = True
    svc = AppService(settings)
    try:
        assert svc.store.count() > 0
        response = svc.search(SearchRequest(query="CE certified 为什么不行", site=Site.EU))
        assert response.results
    finally:
        svc.close()


def test_incremental_ingest_keeps_existing_corpus(service, tmp_path: Path):
    before = len(service.retriever.chunks)
    new_doc = tmp_path / "new_policy.md"
    new_doc.write_text(
        "---\ntitle: 新增测试政策\nsite: US\n---\n# 新增测试政策\n\n## 特殊条款\n\n这是一个用于增量入库测试的独特条款。\n",
        encoding="utf-8",
    )
    stats = service.ingest([new_doc])
    assert stats["total_chunks"] > before
    assert any(chunk.doc_id == "new_policy" for chunk in service.retriever.chunks)
    # Existing documents must remain searchable after the incremental update.
    response = service.search(
        SearchRequest(query="CE certified 为什么不行", site=Site.EU)
    )
    assert response.results
