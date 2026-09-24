"""Embedding providers (offline default + optional neural/API models)."""

from __future__ import annotations

import hashlib
import os
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


class OpenAIEmbedder:
    """Embedding via the OpenAI API (``text-embedding-3-small``).

    Records each call's token usage and cost via the UsageTracker.
    """

    def __init__(self, model: str = "text-embedding-3-small", client=None) -> None:
        self.model = model
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # Lazy import.

            self._client = OpenAI()
        return self._client

    def embed(self, text: str) -> np.ndarray:
        from ..observability.usage_tracker import get_usage_tracker

        tracker = get_usage_tracker()
        with tracker.trace("embed", self.model) as span:
            response = self._get_client().embeddings.create(model=self.model, input=text)
            span.set_usage(input_tokens=response.usage.prompt_tokens)
        return np.asarray(response.data[0].embedding, dtype=float)


def get_embedder() -> Embedder:
    """Return OpenAI (if a key is set), sentence-transformers, else hash."""
    try:
        from dotenv import load_dotenv  # Lazy import.

        load_dotenv()
    except ImportError:
        pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except Exception:  # noqa: BLE001 - fall through on config/network errors
            pass
    try:
        return SentenceTransformerEmbedder()
    except (ImportError, OSError):
        return HashEmbedder()
