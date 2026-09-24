"""BM25 keyword index over chunk texts (rank_bm25 wrapper)."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Sequence

from .schemas import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BM25KeywordIndex:
    """In-memory BM25 index over chunk texts, backed by ``rank_bm25``.

    ``rank_bm25`` does no preprocessing, so tokenization (lowercase + word
    split) is done here for reproducibility.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._chunk_ids: list[str] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase and split on word boundaries; no stemmer (deterministic)."""
        return _TOKEN_RE.findall(text.lower())

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build BM25Okapi over the tokenized chunk texts."""
        if not chunks:
            self._bm25 = None
            self._chunk_ids = []
            return
        from rank_bm25 import BM25Okapi  # Lazy import.

        self._chunk_ids = [c.chunk_id for c in chunks]
        corpus = [self.tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(chunk_id, score)`` pairs, descending."""
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(self.tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            (self._chunk_ids[i], float(scores[i]))
            for i in order[:top_k]
            if scores[i] > 0
        ]

    def save(self, path: str | Path) -> None:
        """Pickle the index for offline reuse."""
        with open(path, "wb") as fh:
            pickle.dump(
                {"k1": self.k1, "b": self.b, "bm25": self._bm25, "chunk_ids": self._chunk_ids},
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25KeywordIndex":
        """Load a pickled index from disk."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        idx = cls(k1=data["k1"], b=data["b"])
        idx._bm25 = data["bm25"]
        idx._chunk_ids = data["chunk_ids"]
        return idx
