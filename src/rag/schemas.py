"""Shared data structures for the retrieval stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A unit of retrievable text.

    Attributes:
        chunk_id: Stable, deterministic identifier (see ``ids.make_chunk_id``).
        source_type: One of "knowledge", "news", or "market_regime".
        source_id: Identifier of the originating document/ticker.
        text: The chunk's text content.
        url: Optional source URL.
        title: Optional source title.
        date: Optional ISO-8601 date string "YYYY-MM-DD" (Chroma-safe primitive).
    """

    chunk_id: str
    source_type: str
    source_id: str
    text: str
    url: str | None = None
    title: str | None = None
    date: str | None = None


@dataclass
class RetrievedChunk(Chunk):
    """A chunk plus its per-ranker fusion evidence."""

    rrf_score: float = 0.0
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
