"""End-to-end backtest + RAG report pipeline.

Ties together data fetch, backtest, hybrid retrieval, reranking, citation
verification, and usage tracking into a single reproducible run. Fully offline
by default (hash embedder + heuristic reranker + in-memory store); plugging in
neural/API models is a drop-in upgrade.
"""

from __future__ import annotations

import logging

from ..backtest import run_backtests
from ..data_loader import fetch_stock_data
from ..evaluate import analyze_failures, compare_strategies
from ..indicators import add_all_indicators
from ..observability.usage_tracker import get_usage_tracker
from ..strategy import MACDStrategy, MAStrategy, CompositeStrategy
from .citation import verify_citations
from .embeddings import HashEmbedder
from .hybrid_retriever import HybridRetriever
from .indexers import build_market_regime_chunks
from .keyword_index import BM25KeywordIndex
from .reranker import get_reranker
from .vector_store import InMemoryStore

logger = logging.getLogger(__name__)

STRATEGIES = {"MA": MAStrategy, "MACD": MACDStrategy, "Composite": CompositeStrategy}
DEFAULT_QUERY_TEMPLATE = "What bullish, bearish, or ranging market regimes occurred for {ticker}?"


def run_report_pipeline(
    ticker: str,
    start: str,
    end: str,
    query: str | None = None,
    top_k: int = 5,
    embed_fn=None,
) -> dict:
    """Run the full pipeline and return a report dict with provenance.

    Args:
        ticker: Yahoo Finance ticker.
        start: Inclusive start date "YYYY-MM-DD".
        end: Inclusive end date "YYYY-MM-DD".
        query: Retrieval query (defaults to a regime-focused question).
        top_k: Number of chunks to retrieve and cite.
        embed_fn: Optional callable ``str -> vector`` (defaults to HashEmbedder).

    Returns:
        A dict with ``report``, ``chunks``, ``citation``, and ``usage`` keys.
    """
    embed_fn = embed_fn or HashEmbedder(dim=128).embed
    tracker = get_usage_tracker()
    tracker.reset()

    df = fetch_stock_data(ticker, start, end)
    indicators = add_all_indicators(df)
    results = run_backtests(df, STRATEGIES)
    comparison = compare_strategies(results)
    failures = {name: analyze_failures(stats) for name, stats in results.items()}

    # Build a market-regime corpus and index it (dense + keyword).
    chunks = build_market_regime_chunks(indicators, ticker)
    keyword = BM25KeywordIndex()
    keyword.index(chunks)
    store = InMemoryStore(embed_fn)
    with tracker.trace("embed", "local:hash") as span:
        store.add(chunks)
        span.set_usage(input_tokens=len(chunks))

    # Hybrid retrieval (dense + BM25 -> RRF) then rerank.
    query_text = query or DEFAULT_QUERY_TEMPLATE.format(ticker=ticker)
    retriever = HybridRetriever(store.query, keyword)
    with tracker.trace("rag.retrieve", "hybrid") as span:
        retrieved = retriever.retrieve(
            query_text, top_k=top_k, chunks_by_id={c.chunk_id: c for c in chunks}
        )
    reranker = get_reranker(prefer_neural=False)
    with tracker.trace("rerank", "local:heuristic") as span:
        top = reranker.rerank(query_text, retrieved, top_k=top_k)

    # Build the report with inline citations, then verify them.
    report_md = _build_report(ticker, start, end, comparison, failures, top)
    citation_map = {i + 1: c.chunk_id for i, c in enumerate(top)}
    retrieved_by_id = {c.chunk_id: c for c in top}
    citation = verify_citations(report_md, citation_map, retrieved_by_id, embed_fn)

    return {
        "ticker": ticker,
        "start": start,
        "end": end,
        "query": query_text,
        "report": report_md,
        "chunks": [
            {
                "id": c.chunk_id,
                "source_type": c.source_type,
                "date": c.date,
                "text": c.text,
                "rerank_score": c.rerank_score,
            }
            for c in top
        ],
        "citation": {
            "coverage": citation.coverage,
            "orphan_rate": citation.orphan_rate,
            "total_citations": citation.total_citations,
            "supported": sum(1 for r in citation.results if r.supported),
        },
        "usage": tracker.to_dict()["totals"],
    }


def _build_report(ticker, start, end, comparison, failures, chunks) -> str:
    lines = [
        f"# Report - {ticker} ({start} to {end})",
        "",
        "## Strategy comparison",
        "",
        comparison.round(2).to_markdown(index=True),
        "",
        "## Failure analysis",
        "",
    ]
    lines += [f"- **{name}**: {text}" for name, text in failures.items()]
    lines += ["", "## Retrieved context", ""]
    lines += [
        f"- [{i}] ({c.source_type}, {c.date or 'n/a'}): {c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    lines += ["", "## Conclusion", "", _conclusion(comparison)]
    return "\n".join(lines)


def _conclusion(comparison) -> str:
    if comparison.empty:
        return "No strategies were evaluated."
    best = comparison["Sharpe Ratio"].idxmax()
    return (
        f"'{best}' posted the highest Sharpe ratio. Past performance is not "
        "indicative of future results."
    )
