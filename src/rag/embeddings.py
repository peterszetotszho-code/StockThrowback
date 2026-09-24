"""Embedding providers (offline default + optional neural/API models)."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .keyword_index import BM25KeywordIndex


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray:
        ...


class HashEmbedder:
    """Deterministic, dependency-free bag-of-hashed-words embedding.

    Used as the offline fallback and in tests. Not semantically strong, but
    reproducible across processes (uses MD5, not Python's randomized ``hash``).
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        for token in BM25KeywordIndex.tokenize(text):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[digest % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class SentenceTransformerEmbedder:
    """Neural embedder via sentence-transformers (lazy import)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # Lazy import.

        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode(text), dtype=float)


def get_embedder() -> Embedder:
    """Return a sentence-transformers embedder if importable, else HashEmbedder."""
    try:
        return SentenceTransformerEmbedder()
    except (ImportError, OSError):
        return HashEmbedder()
