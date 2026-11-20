from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict
from typing import Protocol, Sequence

import httpx
import numpy as np

from ..config import EmbeddingSettings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    model: str

    @property
    def dimension(self) -> int | None: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class _CacheMixin:
    def __init__(self, cache_size: int = 2048):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _cached(self, text: str) -> np.ndarray | None:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        vector = self._cache.get(key)
        if vector is not None:
            self._cache.move_to_end(key)
        return vector

    def _store_cache(self, text: str, vector: np.ndarray) -> None:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


class SiliconFlowEmbedder(_CacheMixin):
    """BGE-M3 through SiliconFlow's OpenAI-compatible /embeddings endpoint."""

    name = "siliconflow"

    def __init__(self, settings: EmbeddingSettings):
        super().__init__(settings.cache_size)
        self.settings = settings
        self.model = settings.model
        self._dimension = settings.dimension
        self._client = httpx.Client(timeout=settings.timeout_s)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension or 1024), dtype=np.float32)
        if not self.settings.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is not configured")

        vectors: list[np.ndarray | None] = [self._cached(text) for text in texts]
        missing = [(i, text) for i, (text, vec) in enumerate(zip(texts, vectors)) if vec is None]
        for start in range(0, len(missing), self.settings.batch_size):
            batch = missing[start : start + self.settings.batch_size]
            payload = {
                "model": self.model,
                "input": [text for _, text in batch],
                "encoding_format": "float",
            }
            data = self._post(payload)
            rows = sorted(data.get("data", []), key=lambda row: row.get("index", 0))
            if len(rows) != len(batch):
                raise RuntimeError(f"embedding count mismatch: {len(rows)} != {len(batch)}")
            matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
            matrix = _normalize(matrix)
            self._dimension = int(matrix.shape[1])
            for (index, text), vector in zip(batch, matrix):
                vectors[index] = vector
                self._store_cache(text, vector)

        stacked = np.asarray(vectors, dtype=np.float32)
        return _normalize(stacked)

    def _post(self, payload: dict) -> dict:
        url = f"{self.settings.base_url.rstrip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(url, headers=headers, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"upstream {response.status_code}")
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.8 * (2**attempt))
        raise RuntimeError(f"SiliconFlow embedding failed: {last_error}")

    def close(self) -> None:
        self._client.close()


class HuggingFaceEmbedder(_CacheMixin):
    """Local sentence-transformers fallback (also works fully offline once cached)."""

    name = "huggingface"

    def __init__(self, settings: EmbeddingSettings):
        super().__init__(settings.cache_size)
        from sentence_transformers import SentenceTransformer

        self.settings = settings
        self.model = settings.model
        self._model = SentenceTransformer(settings.model)
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self._dimension = int(getter())

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension or 768), dtype=np.float32)
        missing = [(i, text) for i, text in enumerate(texts) if self._cached(text) is None]
        if missing:
            encoded = self._model.encode(
                [text for _, text in missing],
                batch_size=self.settings.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for (index, text), vector in zip(missing, encoded):
                vector = np.asarray(vector, dtype=np.float32)
                self._store_cache(text, vector)
        return _normalize(np.asarray([self._cached(t) for t in texts], dtype=np.float32))


class HashEmbedder(_CacheMixin):
    """Deterministic offline embedder for tests/CI; not a semantic model."""

    name = "hash"
    model = "hash-ngram-v1"

    def __init__(self, dimension: int = 512, cache_size: int = 4096):
        super().__init__(cache_size)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            normalized = re.sub(r"\s+", " ", text.lower())
            tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
            grams = tokens + [
                f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)
            ]
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % self._dimension
                sign = 1.0 if (value >> 63) & 1 else -1.0
                matrix[row, index] += sign
        return _normalize(matrix)


def build_embedder(settings: EmbeddingSettings, *, allow_fallback: bool = True) -> Embedder:
    provider = settings.provider.lower()
    if provider == "siliconflow" and settings.api_key:
        return SiliconFlowEmbedder(settings)
    if provider == "huggingface":
        try:
            return HuggingFaceEmbedder(settings)
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise
            logger.warning("HuggingFace embedder unavailable (%s); falling back to hash", exc)
    if provider == "siliconflow":
        logger.warning("SILICONFLOW_API_KEY missing; trying local HF fallback model")
        local = settings.model_copy(update={"model": settings.fallback_model})
        try:
            return HuggingFaceEmbedder(local)
        except Exception as exc:  # noqa: BLE001
            logger.warning("local HF fallback unavailable (%s); using hash embedder", exc)
    return HashEmbedder()
