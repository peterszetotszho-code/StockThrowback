"""Stable chunk identifier generation."""

from __future__ import annotations


def make_chunk_id(source_type: str, source_id: str, chunk_index: int) -> str:
    """Return a stable, deterministic chunk ID.

    Format: ``"{source_type}:{source_id}:{chunk_index:05d}"``. Stability holds
    as long as chunking order is deterministic for a given source.
    """
    return f"{source_type}:{source_id}:{chunk_index:05d}"
