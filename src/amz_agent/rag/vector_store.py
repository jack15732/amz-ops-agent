from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..config import QdrantSettings
from ..models import Chunk

logger = logging.getLogger(__name__)
_NAMESPACE = uuid.UUID("7c1f6a9e-2d4b-4a6f-9a4e-2f6d8c1b3a50")


class VectorStore(Protocol):
    def ensure(self, dimension: int) -> None: ...

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self, vector: np.ndarray, top_k: int, site: str | None = None
    ) -> list[tuple[Chunk, float]]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def all_chunks(self) -> list[Chunk]: ...


class MemoryVectorStore:
    """Deterministic in-memory store used by tests and the eval harness."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def ensure(self, dimension: int) -> None:
        return None

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = np.asarray(vector, dtype=np.float32)

    def search(
        self, vector: np.ndarray, top_k: int, site: str | None = None
    ) -> list[tuple[Chunk, float]]:
        scored: list[tuple[Chunk, float]] = []
        for chunk_id, chunk in self._chunks.items():
            if site and chunk.metadata.get("site", "ALL") not in {site, "ALL"}:
                continue
            score = float(np.dot(vector, self._vectors[chunk_id]))
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
        self._vectors.clear()

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())


class QdrantVectorStore:
    def __init__(self, settings: QdrantSettings):
        from qdrant_client import QdrantClient

        self.settings = settings
        if settings.mode == "server":
            self.client = QdrantClient(url=settings.url, api_key=settings.api_key or None)
        elif settings.mode == "memory":
            self.client = QdrantClient(location=":memory:")
        else:
            path = Path(settings.path) if settings.path else None
            if path is None:
                raise ValueError("qdrant local mode requires settings.path")
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        self.collection = settings.collection
        self._dimension: int | None = None

    def ensure(self, dimension: int) -> None:
        from qdrant_client.http.models import Distance, VectorParams

        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            current = info.config.params.vectors.size
            if current == dimension:
                self._dimension = dimension
                return
            logger.warning(
                "collection %s dim %s != %s, recreating", self.collection, current, dimension
            )
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        self._dimension = dimension

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.http.models import PointStruct

        if not len(chunks):
            return
        self.ensure(int(vectors.shape[1]))
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = chunk.model_dump()
            payload["site"] = chunk.metadata.get("site", "ALL")
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(_NAMESPACE, chunk.chunk_id)),
                    vector=np.asarray(vector, dtype=float).tolist(),
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(
        self, vector: np.ndarray, top_k: int, site: str | None = None
    ) -> list[tuple[Chunk, float]]:
        if not self.client.collection_exists(self.collection):
            return []
        query_filter = None
        if site:
            from qdrant_client.http.models import FieldCondition, Filter, MatchAny

            query_filter = Filter(
                must=[FieldCondition(key="site", match=MatchAny(any=[site, "ALL"]))]
            )
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=np.asarray(vector, dtype=float).tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            points = response.points
        else:  # pragma: no cover - older qdrant-client
            points = self.client.search(
                collection_name=self.collection,
                query_vector=np.asarray(vector, dtype=float).tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        results: list[tuple[Chunk, float]] = []
        for point in points:
            payload = dict(point.payload or {})
            payload.pop("site", None)
            try:
                results.append((Chunk.model_validate(payload), float(point.score)))
            except Exception:  # noqa: BLE001
                continue
        return results

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._dimension = None

    def all_chunks(self) -> list[Chunk]:
        if not self.client.collection_exists(self.collection):
            return []
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                payload.pop("site", None)
                try:
                    chunks.append(Chunk.model_validate(payload))
                except Exception:  # noqa: BLE001
                    continue
            if offset is None:
                break
        return chunks


def build_vector_store(settings: QdrantSettings, data_dir: Path) -> VectorStore:
    if settings.mode == "memory":
        return MemoryVectorStore()
    settings = settings.model_copy(update={"path": str(data_dir / "qdrant")})
    return QdrantVectorStore(settings)
