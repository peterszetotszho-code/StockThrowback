"""ChromaDB-backed vector store (lazy import, offline-friendly)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

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
