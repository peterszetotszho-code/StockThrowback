"""Reranking of fused candidates (neural cross-encoder + heuristic fallback)."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Protocol, Sequence

from .keyword_index import BM25KeywordIndex
from .schemas import Chunk, RetrievedChunk

_AUTHORITY = {"knowledge": 0.9, "news": 0.6, "market_regime": 0.7}
_DEFAULT_AUTHORITY = 0.5


class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int = 10) -> list[RetrievedChunk]:
        ...


class HeuristicReranker:
    """Offline fallback: keyword overlap + recency + source authority."""

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int = 10) -> list[RetrievedChunk]:
        scored = [(self._score(query, c), c) for c in chunks]
        scored.sort(key=lambda pair: -pair[0])
        return [c for _, c in scored[:top_k]]

    def _score(self, query: str, chunk: RetrievedChunk) -> float:
        return (
            0.55 * self.keyword_overlap(query, chunk.text)
            + 0.25 * self.recency(chunk)
            + 0.20 * self.authority(chunk)
        )

    @staticmethod
    def keyword_overlap(query: str, text: str) -> float:
        """Fraction of query terms that appear in the text, in [0, 1]."""
        query_terms = set(BM25KeywordIndex.tokenize(query))
        if not query_terms:
            return 0.0
        text_terms = set(BM25KeywordIndex.tokenize(text))
        return len(query_terms & text_terms) / len(query_terms)

    @staticmethod
    def recency(chunk: Chunk, now: datetime | None = None, half_life_days: float = 180.0) -> float:
        """Exponential decay by chunk age; 0.5 when the date is unknown."""
        if not chunk.date:
            return 0.5
        try:
            date = datetime.fromisoformat(chunk.date)
        except ValueError:
            return 0.5
        now = now or datetime.now()
        age_days = max(0.0, (now - date).days)
        return math.exp(-age_days / half_life_days)

    @staticmethod
    def authority(chunk: Chunk) -> float:
        """Static per-source-type prior (reproducible, not learned)."""
        return _AUTHORITY.get(chunk.source_type, _DEFAULT_AUTHORITY)


class CrossEncoderReranker:
    """Neural reranker via sentence-transformers ``CrossEncoder``."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        from sentence_transformers import CrossEncoder  # Lazy import (pulls torch).

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int = 10) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks])
        ranked = sorted(zip(scores, chunks), key=lambda pair: -pair[0])
        out = []
        for score, chunk in ranked[:top_k]:
            chunk.rerank_score = float(score)
            out.append(chunk)
        return out


def get_reranker(prefer_neural: bool = True) -> Reranker:
    """Return ``CrossEncoderReranker`` if importable, else ``HeuristicReranker``."""
    if prefer_neural:
        try:
            return CrossEncoderReranker()
        except (ImportError, OSError):
            return HeuristicReranker()
    return HeuristicReranker()
