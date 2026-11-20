"""Small BM25 implementation with a CJK-friendly tokenizer.

The corpus is only a few thousand chunks, so an in-process index is cheaper and
more transparent than another service. Tokens combine latin words, jieba words
and CJK bigrams, which keeps recall for domain abbreviations and Japanese text.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Sequence

from ..models import Chunk

try:  # jieba is optional at runtime; bigrams still work without it
    import jieba

    jieba.setLogLevel(60)
except Exception:  # pragma: no cover
    jieba = None

_LATIN_RE = re.compile(r"[a-z0-9][a-z0-9+#._-]*")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "for", "with", "in", "on", "is",
    "are", "be", "as", "at", "by", "this", "that", "it", "from", "can", "will",
    "的", "了", "和", "是", "在", "有", "与", "及", "对", "吗", "呢", "怎么",
    "什么", "如何", "是否", "可以", "需要", "一个", "这个", "那个",
}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _LATIN_RE.finditer(text):
        token = match.group(0).strip("._-")
        if len(token) >= 2 and token not in _STOP:
            tokens.append(token)
    for block in _CJK_RE.findall(text):
        if jieba is not None:
            tokens.extend(
                token.strip()
                for token in jieba.cut(block)
                if len(token.strip()) >= 2 and token.strip() not in _STOP
            )
        # Bigrams keep recall when the segmenter does not know the domain term.
        if len(block) == 1:
            if block not in _STOP:
                tokens.append(block)
        else:
            tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
    return tokens


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = list(chunks)
        self.doc_tokens: list[list[str]] = [tokenize(c.embed_text) for c in self.chunks]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        total = len(self.chunks)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(
        self,
        query_tokens: Sequence[str],
        top_k: int = 20,
        *,
        site: str | None = None,
    ) -> list[tuple[Chunk, float, list[str]]]:
        if not self.chunks or not query_tokens:
            return []
        query = list(dict.fromkeys(query_tokens))
        scores: list[float] = [0.0] * len(self.chunks)
        for index, chunk in enumerate(self.chunks):
            if site and chunk.metadata.get("site", "ALL") not in {site, "ALL"}:
                continue
            freqs = self.term_freqs[index]
            dl = self.doc_len[index] or 1
            score = 0.0
            for term in query:
                tf = freqs.get(term)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores[index] = score
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        output: list[tuple[Chunk, float, list[str]]] = []
        for index in ranked[:top_k]:
            if scores[index] <= 0:
                break
            matched = [term for term in query if term in self.term_freqs[index]][:8]
            output.append((self.chunks[index], scores[index], matched))
        return output
