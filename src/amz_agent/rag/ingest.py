from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import yaml

from ..config import PROJECT_ROOT
from ..models import Chunk
from .chunker import Chunker, Section
from .embeddings import Embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".docx", ".xlsx"}


def _source(path: Path) -> str:
    """Keep citation paths portable inside the repository."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _slug(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", path.stem).strip("-") or "doc"


def _read_markdown(path: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            raw = parts[2]

    lines = raw.splitlines()
    title = str(meta.get("title") or path.stem)
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if not text:
            return
        path_titles = [t for _, t in stack] or [title]
        sections.append(
            Section(
                doc_id=_slug(path),
                title=title,
                source=_source(path),
                source_type="markdown",
                heading_path=path_titles,
                text=text,
                metadata=dict(meta),
            )
        )
        buffer.clear()

    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        heading = None if in_fence else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            stack = [(lvl, txt) for lvl, txt in stack if lvl < level]
            stack.append((level, text))
            if level == 1 and title == path.stem:
                title = text
        else:
            buffer.append(line)
    flush()

    if not sections:
        text = raw.strip()
        if text:
            sections.append(
                Section(
                    doc_id=_slug(path),
                    title=title,
                    source=_source(path),
                    source_type="markdown",
                    heading_path=[title],
                    text=text,
                    metadata=dict(meta),
                )
            )
    return sections


def _read_docx(path: Path) -> list[Section]:
    from docx import Document

    doc = Document(str(path))
    title = path.stem
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                Section(
                    doc_id=_slug(path),
                    title=title,
                    source=_source(path),
                    source_type="docx",
                    heading_path=[t for _, t in stack] or [title],
                    text=text,
                )
            )
            buffer.clear()

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading"):
            match = re.search(r"(\d+)", style)
            level = int(match.group(1)) if match else 1
            flush()
            stack = [(lvl, txt) for lvl, txt in stack if lvl < level]
            stack.append((level, text))
        else:
            buffer.append(text)
    for table in doc.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        buffer.append("\n".join(r for r in rows if r.strip()))
    flush()
    return sections


def _read_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections: list[Section] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(
                Section(
                    doc_id=_slug(path),
                    title=path.stem,
                    source=_source(path),
                    source_type="pdf",
                    heading_path=[path.stem, f"第 {number} 页"],
                    text=text,
                    page=number,
                )
            )
    return sections


def _read_xlsx(path: Path) -> list[Section]:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    sections: list[Section] = []
    for sheet in workbook.worksheets:
        rows: list[tuple[int, str]] = []
        for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
            if cells:
                rows.append((row_index, " | ".join(cells)))
        for start in range(0, len(rows), 15):
            block = rows[start : start + 15]
            sections.append(
                Section(
                    doc_id=_slug(path),
                    title=path.stem,
                    source=_source(path),
                    source_type="xlsx",
                    heading_path=[path.stem, f"工作表：{sheet.title}"],
                    text="\n".join(text for _, text in block),
                    sheet=sheet.title,
                    row_start=block[0][0],
                    row_end=block[-1][0],
                )
            )
    workbook.close()
    return sections


def read_sections(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _read_markdown(path)
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8").strip()
        return (
            [
                Section(
                    doc_id=_slug(path),
                    title=path.stem,
                    source=_source(path),
                    source_type="text",
                    heading_path=[path.stem],
                    text=text,
                )
            ]
            if text
            else []
        )
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".xlsx":
        return _read_xlsx(path)
    raise ValueError(f"unsupported file type: {path.suffix}")


def collect_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(
                p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED:
            files.append(path)
    return files


def build_chunks(paths: Iterable[str | Path], chunker: Chunker) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for file in collect_files(paths):
        try:
            sections = read_sections(file)
        except Exception as exc:  # noqa: BLE001 - one bad file should not abort a batch
            logger.warning("skip %s: %s", file, exc)
            continue
        count = 0
        for section in sections:
            for chunk in chunker.split(section):
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                chunks.append(chunk)
                count += 1
        logger.info("parsed %s -> %d chunks", file.name, count)
    return chunks


def ingest_paths(
    paths: Iterable[str | Path],
    embedder: Embedder,
    store: VectorStore,
    *,
    chunker: Chunker,
    catalog_path: Path,
    reset: bool = False,
    batch_size: int = 32,
) -> dict:
    new_chunks = build_chunks(paths, chunker)
    if not new_chunks:
        return {"files": 0, "chunks": 0, "total_chunks": len(load_catalog(catalog_path))}

    # Incremental ingestion must not drop the existing catalog/BM25 corpus.
    merged: dict[str, Chunk] = {}
    if not reset:
        merged.update({chunk.chunk_id: chunk for chunk in load_catalog(catalog_path)})
    merged.update({chunk.chunk_id: chunk for chunk in new_chunks})
    all_chunks = list(merged.values())

    if reset:
        store.reset()
    for start in range(0, len(new_chunks), batch_size):
        batch = new_chunks[start : start + batch_size]
        vectors = embedder.embed([chunk.embed_text for chunk in batch])
        store.upsert(batch, vectors)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")

    return {
        "files": len({chunk.source for chunk in new_chunks}),
        "chunks": len(new_chunks),
        "embedded": len(new_chunks),
        "total_chunks": len(all_chunks),
        "docs": len({chunk.doc_id for chunk in all_chunks}),
    }


def load_catalog(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                chunks.append(Chunk.model_validate_json(line))
            except Exception:  # noqa: BLE001
                continue
    return chunks
