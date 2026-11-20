"""Heading-aware chunking with a character sliding window.

Each chunk keeps the full heading path and source coordinates, which is what
makes citations ("doc > section > subsection") reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..models import Chunk

_BREAK_CHARS = "。！？.!?；;\n"


@dataclass
class Section:
    doc_id: str
    title: str
    source: str
    source_type: str
    heading_path: list[str] = field(default_factory=list)
    text: str = ""
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker:
    def __init__(self, max_chars: int = 760, overlap: int = 120):
        if max_chars < 120:
            raise ValueError("max_chars must be >= 120")
        if overlap < 0 or overlap >= max_chars:
            raise ValueError("overlap must be in [0, max_chars)")
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, section: Section) -> list[Chunk]:
        text = _clean(section.text)
        if not text:
            return []
        windows = self._windows(text)
        chunks: list[Chunk] = []
        for index, window in enumerate(windows):
            chunks.append(
                Chunk(
                    chunk_id="",  # filled below once we know all fields
                    doc_id=section.doc_id,
                    title=section.title,
                    source=section.source,
                    source_type=section.source_type,
                    heading_path=list(section.heading_path),
                    text=window,
                    page=section.page,
                    sheet=section.sheet,
                    row_start=section.row_start,
                    row_end=section.row_end,
                    metadata=dict(section.metadata),
                )
            )
            chunks[-1].chunk_id = Chunk.make_id(
                section.doc_id, section.heading_path, index, window
            )
        return chunks

    def _windows(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        windows: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            hard_end = min(length, start + self.max_chars)
            end = hard_end
            if hard_end < length:
                search_from = start + int(self.max_chars * 0.55)
                for pos in range(hard_end, search_from, -1):
                    if text[pos - 1] in _BREAK_CHARS:
                        end = pos
                        break
            window = text[start:end].strip()
            if window:
                windows.append(window)
            if end >= length:
                break
            start = max(start + 1, end - self.overlap)
        return windows


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
