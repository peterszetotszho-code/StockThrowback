"""End-to-end backtest + RAG report pipeline.

Ties together data fetch, backtest, hybrid retrieval, reranking, news event
recall, citation verification, and usage tracking into a single reproducible
run. Fully offline by default (hash embedder + heuristic reranker + in-memory
store); plugging in neural/API models is a drop-in upgrade.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..backtest import run_backtests
from ..data_loader import fetch_stock_data
from ..evaluate import analyze_failures, compare_strategies
from ..indicators import add_all_indicators
from ..observability.usage_tracker import get_usage_tracker
from ..strategy import MACDStrategy, MAStrategy, CompositeStrategy
from .citation import verify_citations
from .embeddings import HashEmbedder
from .hybrid_retriever import HybridRetriever
from .indexers import build_market_regime_chunks, build_news_chunks, fetch_gdelt_articles
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
    news_query: str | None = None,
    max_news: int = 10,
) -> dict:
    """Run the full pipeline and return a report dict with provenance.

    Args:
        ticker: Yahoo Finance ticker.
        start: Inclusive start date "YYYY-MM-DD".
        end: Inclusive end date "YYYY-MM-DD".
        query: Retrieval query (defaults to a regime-focused question).
        top_k: Number of market-regime chunks to retrieve and cite.
        embed_fn: Optional callable ``str -> vector`` (defaults to HashEmbedder).
        news_query: Optional company name used to recall news for the worst
            drawdown window (network required).
        max_news: Maximum news articles to recall.

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

    # Market-regime corpus + hybrid retrieval.
    regime_chunks = build_market_regime_chunks(indicators, ticker)
    keyword = BM25KeywordIndex()
    keyword.index(regime_chunks)
    store = InMemoryStore(embed_fn)
    with tracker.trace("embed", "local:hash") as span:
        store.add(regime_chunks)
        span.set_usage(input_tokens=len(regime_chunks))

    query_text = query or DEFAULT_QUERY_TEMPLATE.format(ticker=ticker)
    retriever = HybridRetriever(store.query, keyword)
    with tracker.trace("rag.retrieve", "hybrid") as span:
        retrieved = retriever.retrieve(
            query_text, top_k=top_k, chunks_by_id={c.chunk_id: c for c in regime_chunks}
        )
    reranker = get_reranker(prefer_neural=False)
    with tracker.trace("rerank", "local:heuristic") as span:
        top = reranker.rerank(query_text, retrieved, top_k=top_k)

    # Event recall: news during the worst drawdown window (direct, not retrieved).
    dd_start, dd_end = _worst_drawdown_window(results, start, end)
    news_chunks = []
    if news_query:
        try:
            articles = fetch_gdelt_articles(news_query, dd_start, dd_end, max_records=max_news)
            news_chunks = build_news_chunks(articles)
            logger.info(
                "Recalled %d news articles for %r (%s to %s).",
                len(news_chunks), news_query, dd_start, dd_end,
            )
        except Exception as exc:  # noqa: BLE001 - network failures are non-fatal
            logger.warning("News recall skipped: %s", exc)

    report_md = _build_report(
        ticker, start, end, comparison, failures, top, news_chunks, dd_start, dd_end, news_query
    )

    cited = list(top) + news_chunks
    citation_map = {i + 1: c.chunk_id for i, c in enumerate(cited)}
    citation = verify_citations(report_md, citation_map, {c.chunk_id: c for c in cited}, embed_fn)

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
                "rerank_score": getattr(c, "rerank_score", None),
            }
            for c in cited
        ],
        "citation": {
            "coverage": citation.coverage,
            "orphan_rate": citation.orphan_rate,
            "total_citations": citation.total_citations,
            "supported": sum(1 for r in citation.results if r.supported),
        },
        "usage": tracker.to_dict()["totals"],
    }


def _worst_drawdown_window(results: dict, default_start: str, default_end: str) -> tuple[str, str]:
    """Return the (start, end) "YYYY-MM-DD" window around the deepest drawdown."""
    best_date = None
    best_depth = 0.0
    for stats in results.values():
        equity = getattr(stats, "_equity_curve", None)
        if equity is None or equity.empty or "DrawdownPct" not in equity.columns:
            continue
        drawdown = equity["DrawdownPct"]
        depth = float(drawdown.max())
        if depth > best_depth:
            best_depth = depth
            best_date = drawdown.idxmax()
    if best_date is None:
        return default_start, default_end
    window_start = (best_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    window_end = (best_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    return window_start, window_end


def _build_report(ticker, start, end, comparison, failures, regime_chunks, news_chunks, dd_start, dd_end, news_query) -> str:
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

    lines += ["", "## Market regime context", ""]
    lines += [
        f"- [{i}] ({c.source_type}, {c.date or 'n/a'}): {c.text}"
        for i, c in enumerate(regime_chunks, start=1)
    ]

    lines += ["", f"## News during worst drawdown ({dd_start} to {dd_end})", ""]
    if news_chunks:
        offset = len(regime_chunks)
        lines += [
            f"- [{offset + i}] ({c.date or 'n/a'}): {c.text}"
            for i, c in enumerate(news_chunks, start=1)
        ]
    elif news_query:
        lines.append("_No news recalled for this period._")
    else:
        lines.append("_News recall disabled (no company query provided)._")

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
