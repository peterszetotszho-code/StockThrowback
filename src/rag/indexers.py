"""Chunk builders for the three RAG sources: knowledge, news, market regime."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .chunking import chunk_text
from .ids import make_chunk_id
from .schemas import Chunk

logger = logging.getLogger(__name__)


def build_knowledge_chunks(
    directory: str | Path, chunk_size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Build chunks from ``.md`` / ``.txt`` files under ``directory``."""
    directory = Path(directory)
    if not directory.exists():
        logger.warning("Knowledge directory %s does not exist.", directory)
        return []
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")) + sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        source_id = path.stem
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id("knowledge", source_id, i),
                    source_type="knowledge",
                    source_id=source_id,
                    text=piece,
                    title=path.name,
                )
            )
    return chunks


def build_news_chunks(articles: list[dict]) -> list[Chunk]:
    """Build chunks from a list of article dicts (title/url/date/source_id)."""
    chunks: list[Chunk] = []
    for i, article in enumerate(articles):
        title = article.get("title", "")
        if not title:
            continue
        source_id = article.get("source_id", f"news_{i}")
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id("news", source_id, 0),
                source_type="news",
                source_id=source_id,
                text=title,
                title=title,
                url=article.get("url"),
                date=article.get("date"),
            )
        )
    return chunks


def fetch_gdelt_articles(query: str, start: str, end: str, max_records: int = 25) -> list[dict]:
    """Fetch historical news headlines from GDELT (free, no API key).

    Returns a list of dicts with keys ``title``, ``url``, ``date``,
    ``source_id``. Only headlines are available in ``artlist`` mode (no full
    text), which is an honest limitation of the free source.
    """
    import requests  # Local import.

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": f"{start}000000",
        "enddatetime": f"{end}235959",
        "maxrecords": max_records,
        "sort": "datedesc",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    out = []
    for i, article in enumerate(articles):
        out.append(
            {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "date": (article.get("seendate", "") or "")[:8],
                "source_id": f"gdelt_{i}",
            }
        )
    return out


def build_market_regime_chunks(
    df: pd.DataFrame, ticker: str, step: int = 5
) -> list[Chunk]:
    """Describe rolling market conditions as retrievable text chunks.

    ``df`` is expected to already carry indicator columns (e.g. via
    ``indicators.add_all_indicators``).
    """
    chunks: list[Chunk] = []
    for i in range(0, len(df), step):
        row = df.iloc[i]
        date = df.index[i]
        text = (
            f"{date:%Y-%m-%d} {ticker} {row.get('Trend_State', 'ranging')} regime: "
            f"Close={row['Close']:.2f}, RSI={row.get('RSI', float('nan')):.1f}, "
            f"MA5={row.get('MA5', float('nan')):.2f} vs "
            f"MA20={row.get('MA20', float('nan')):.2f}, "
            f"MACD={row.get('MACD', float('nan')):.2f}, Volume={row['Volume']}"
        )
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id("market_regime", ticker, i),
                source_type="market_regime",
                source_id=ticker,
                text=text,
                date=date.strftime("%Y-%m-%d"),
            )
        )
    return chunks
