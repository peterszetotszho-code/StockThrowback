"""ChromaDB-backed vector store (lazy import, offline-friendly)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .schemas import Chunk

logger = logging.getLogger(__name__)


class ChromaStore:
    """Thin wrapper over a ChromaDB collection for chunk storage and query.

    ``embed_fn`` is expected (a local or API embedder) so that Chroma's default
    ONNX model is never implicitly downloaded. ``persist_dir`` enables on-disk
    persistence; omit it for an in-memory store (tests).
    """

    def __init__(
        self,
        collection_name: str = "stock-trend-lab",
        persist_dir: str | Path | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        import chromadb  # Lazy import.

        self._embed = embed_fn
        if persist_dir:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(name=collection_name)

    @staticmethod
    def _meta(chunk: Chunk) -> dict:
        meta: dict = {"source_type": chunk.source_type, "source_id": chunk.source_id}
        if chunk.url:
            meta["url"] = chunk.url
        if chunk.title:
            meta["title"] = chunk.title
        if chunk.date:
            meta["date"] = chunk.date
        return meta

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the collection with their metadata."""
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._meta(c) for c in chunks]
        if self._embed is not None:
            embeddings = [self._embed(c.text) for c in chunks]
            self._collection.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )
        else:
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return ``(chunk_id, similarity)`` pairs, descending by similarity."""
        if self._embed is not None:
            res = self._collection.query(
                query_embeddings=[self._embed(query)],
                n_results=top_k,
                include=["distances"],
            )
        else:
            res = self._collection.query(
                query_texts=[query], n_results=top_k, include=["distances"]
            )
        ids = res["ids"][0]
        distances = res["distances"][0]
        # Chroma returns a distance (lower = closer); convert to similarity.
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, distances)]


class InMemoryStore:
    """Dependency-free dense retriever using cosine similarity over embeddings.

    A lightweight fallback to ``ChromaStore`` for offline use and demos; it
    keeps the same ``(chunk_id, similarity)`` query contract.
    """

    def __init__(self, embed_fn: Callable[[str], list[float] | np.ndarray]) -> None:
        self._embed = embed_fn
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._ids.append(chunk.chunk_id)
            self._vectors.append(np.asarray(self._embed(chunk.text), dtype=float))

    def query(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        matrix = np.stack(self._vectors)
        q = np.asarray(self._embed(query), dtype=float)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        similarities = (matrix @ q) / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-similarities)[:top_k]
        return [(self._ids[i], float(similarities[i])) for i in order]
