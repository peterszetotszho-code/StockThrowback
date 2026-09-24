"""Hybrid dense + sparse retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Callable, Sequence

from .schemas import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked ``(doc_id, score)`` lists with Reciprocal Rank Fusion.

    ``score(d) = sum over lists of 1 / (k + rank_r(d))``; a document absent
    from a list contributes 0 from that list. Returns ``(doc_id, rrf_score)``
    sorted descending, ties broken by ``doc_id`` ascending for determinism.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    """Combine a dense retriever and a BM25 keyword index via RRF."""

    def __init__(
        self,
        dense_fn: Callable[[str, int], list[tuple[str, float]]] | None,
        keyword_index,
        k: int = 60,
    ) -> None:
        self._dense_fn = dense_fn
        self._keyword = keyword_index
        self._k = k

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        chunks_by_id: dict[str, Chunk] | None = None,
    ) -> list[RetrievedChunk]:
        """Run both retrievers, fuse with RRF, and enrich with metadata.

        Args:
            query: The natural-language query.
            top_k: Number of fused results to return.
            chunks_by_id: Optional ``chunk_id -> Chunk`` map for enrichment.

        Returns:
            A list of ``RetrievedChunk`` with per-ranker rank/score metadata.
        """
        dense = self._dense_fn(query, top_k) if self._dense_fn else []
        sparse = self._keyword.search(query, top_k) if self._keyword else []
        fused = reciprocal_rank_fusion([dense, sparse], k=self._k)[:top_k]

        dense_rank = {doc_id: i + 1 for i, (doc_id, _) in enumerate(dense)}
        dense_score = dict(dense)
        bm25_rank = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sparse)}
        bm25_score = dict(sparse)
        lookup = chunks_by_id or {}

        results: list[RetrievedChunk] = []
        for doc_id, rrf in fused:
            chunk = lookup.get(doc_id)
            results.append(
                RetrievedChunk(
                    chunk_id=doc_id,
                    source_type=chunk.source_type if chunk else "unknown",
                    source_id=chunk.source_id if chunk else "",
                    text=chunk.text if chunk else "",
                    url=chunk.url if chunk else None,
                    title=chunk.title if chunk else None,
                    date=chunk.date if chunk else None,
                    rrf_score=rrf,
                    dense_rank=dense_rank.get(doc_id),
                    dense_score=dense_score.get(doc_id),
                    bm25_rank=bm25_rank.get(doc_id),
                    bm25_score=bm25_score.get(doc_id),
                )
            )
        return results
