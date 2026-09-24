"""End-to-end backtest + RAG report pipeline.

Ties together data fetch, backtest, hybrid retrieval (market-regime +
knowledge corpora), reranking, news event recall, citation verification, and
usage tracking into a single reproducible run. Fully offline by default (hash
embedder + heuristic reranker + in-memory store); plugging in neural/API models
is a drop-in upgrade.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from ..agent.llm import OpenAIChatModel
from ..backtest import run_backtests
from ..data_loader import fetch_stock_data
from ..evaluate import analyze_failures, compare_strategies
from ..indicators import add_all_indicators
from ..observability.usage_tracker import get_usage_tracker
from ..strategy import MACDStrategy, MAStrategy, CompositeStrategy
from .citation import verify_citations
from .embeddings import HashEmbedder
from .hybrid_retriever import HybridRetriever
from .indexers import (
    build_knowledge_chunks,
    build_market_regime_chunks,
    build_news_chunks,
    fetch_gdelt_articles,
)
from .keyword_index import BM25KeywordIndex
from .reranker import get_reranker
from .vector_store import InMemoryStore

logger = logging.getLogger(__name__)

STRATEGIES = {"MA": MAStrategy, "MACD": MACDStrategy, "Composite": CompositeStrategy}
DEFAULT_QUERY_TEMPLATE = "What bullish, bearish, or ranging market regimes occurred for {ticker}?"
KNOWLEDGE_QUERY = "failure modes and limitations of moving average and trend strategies"
DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge"

REPORT_PROMPT = """\
Write a concise Markdown report for {ticker} ({start} to {end}) using ONLY the
facts below. Cite retrieved chunks inline with their [n] number. Do not invent
numbers. Include sections: Strategy comparison, Failure analysis, Market regime
context, Knowledge context, News, and Conclusion.

Strategy comparison:
{comparison}

Failure analysis:
{failures}

Retrieved chunks (cite with [n]):
{chunks}
"""


def run_report_pipeline(
    ticker: str,
    start: str,
    end: str,
    query: str | None = None,
    top_k: int = 5,
    embed_fn=None,
    news_query: str | None = None,
    max_news: int = 10,
    knowledge_dir: str | Path | None = None,
    llm=None,
) -> dict:
    """Run the full pipeline and return a report dict with provenance.

    Args:
        ticker: Yahoo Finance ticker.
        start: Inclusive start date "YYYY-MM-DD".
        end: Inclusive end date "YYYY-MM-DD".
        query: Market-regime retrieval query (defaults to a regime question).
        top_k: Number of chunks to retrieve and cite per corpus.
        embed_fn: Optional callable ``str -> vector`` (defaults to HashEmbedder).
        news_query: Optional company name used to recall news for the worst
            drawdown window (network required).
        max_news: Maximum news articles to recall.
        knowledge_dir: Directory of knowledge docs (defaults to ``knowledge/``).

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

    query_text = query or DEFAULT_QUERY_TEMPLATE.format(ticker=ticker)

    # Market-regime retrieval.
    regime_chunks = build_market_regime_chunks(indicators, ticker)
    top_regime = _retrieve(regime_chunks, query_text, top_k, embed_fn, tracker)

    # Knowledge retrieval (only if docs are present).
    knowledge_dir = Path(knowledge_dir) if knowledge_dir else DEFAULT_KNOWLEDGE_DIR
    knowledge_chunks = build_knowledge_chunks(knowledge_dir)
    top_knowledge = (
        _retrieve(knowledge_chunks, KNOWLEDGE_QUERY, min(top_k, len(knowledge_chunks)), embed_fn, tracker)
        if knowledge_chunks
        else []
    )

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

    cited = list(top_regime) + list(top_knowledge) + news_chunks
    model = llm if llm is not None else _auto_llm()
    if model is not None:
        try:
            report_md = _generate_report_with_llm(model, ticker, start, end, comparison, failures, cited)
            if not report_md:
                raise ValueError("empty LLM report")
        except Exception as exc:  # noqa: BLE001 - fall back to the template on any LLM failure
            logger.warning("LLM report failed (%s); using template report.", exc)
            report_md = _build_report(
                ticker, start, end, comparison, failures, top_regime, top_knowledge,
                news_chunks, dd_start, dd_end, news_query,
            )
    else:
        report_md = _build_report(
            ticker, start, end, comparison, failures, top_regime, top_knowledge,
            news_chunks, dd_start, dd_end, news_query,
        )

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


def _retrieve(chunks, query: str, top_k: int, embed_fn, tracker):
    """Index ``chunks`` (dense + BM25), retrieve, rerank, and return the top-k."""
    keyword = BM25KeywordIndex()
    keyword.index(chunks)
    store = InMemoryStore(embed_fn)
    with tracker.trace("embed", "local:hash") as span:
        store.add(chunks)
        span.set_usage(input_tokens=len(chunks))
    retriever = HybridRetriever(store.query, keyword)
    with tracker.trace("rag.retrieve", "hybrid") as span:
        retrieved = retriever.retrieve(query, top_k=top_k, chunks_by_id={c.chunk_id: c for c in chunks})
    reranker = get_reranker(prefer_neural=False)
    with tracker.trace("rerank", "local:heuristic") as span:
        return reranker.rerank(query, retrieved, top_k=top_k)


def _auto_llm():
    """Return an OpenAIChatModel if an API key is configured, else None."""
    try:
        from dotenv import load_dotenv  # Lazy import.

        load_dotenv()
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    try:
        return OpenAIChatModel(model=model)
    except Exception as exc:  # noqa: BLE001 - config errors
        logger.warning("LLM unavailable (%s); using template report.", exc)
        return None


def _generate_report_with_llm(llm, ticker, start, end, comparison, failures, cited) -> str:
    """Generate the report with an LLM, grounded on the retrieved chunks."""
    chunks_text = "\n".join(
        f"[{i}] ({c.source_type}): {c.text}" for i, c in enumerate(cited, start=1)
    )
    failures_text = "\n".join(f"- **{name}**: {text}" for name, text in failures.items())
    prompt = REPORT_PROMPT.format(
        ticker=ticker,
        start=start,
        end=end,
        comparison=comparison.round(2).to_markdown(index=True),
        failures=failures_text,
        chunks=chunks_text,
    )
    response = llm.complete([{"role": "user", "content": prompt}], temperature=0.0)
    return response.content or ""


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


def _build_report(
    ticker, start, end, comparison, failures,
    regime_chunks, knowledge_chunks, news_chunks, dd_start, dd_end, news_query,
) -> str:
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

    offset = len(regime_chunks)
    if knowledge_chunks:
        lines += ["", "## Knowledge context", ""]
        lines += [
            f"- [{offset + i}] ({c.source_type}, {c.title or 'n/a'}): {c.text}"
            for i, c in enumerate(knowledge_chunks, start=1)
        ]
        offset += len(knowledge_chunks)

    lines += ["", f"## News during worst drawdown ({dd_start} to {dd_end})", ""]
    if news_chunks:
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
