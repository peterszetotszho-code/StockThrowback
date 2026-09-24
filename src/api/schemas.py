"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

STRATEGY_NAMES = ("MA", "MACD", "Composite")


class BacktestRequest(BaseModel):
    """Request body for the backtest endpoint."""

    ticker: str = "0700.HK"
    start: str = "2022-01-01"
    end: str = "2024-12-31"
    strategies: list[str] = Field(default_factory=lambda: list(STRATEGY_NAMES))
    cash: float = 100000.0
    commission: float = 0.001


class Candle(BaseModel):
    """One OHLCV bar plus its computed indicators."""

    time: int  # unix seconds (UTC), for lightweight-charts
    open: float
    high: float
    low: float
    close: float
    volume: int
    ma5: float | None = None
    ma20: float | None = None
    ma60: float | None = None
    rsi: float | None = None
    trend: str = "ranging"


class EquityPoint(BaseModel):
    """One point of an equity curve (equity + drawdown as negative percent)."""

    time: int
    equity: float
    drawdown: float


class BacktestResponse(BaseModel):
    """Response body for the backtest endpoint."""

    ticker: str
    company_name: str | None = None
    currency: str | None = None
    start: str
    end: str
    candles: list[Candle]
    comparison: list[dict]
    failures: dict[str, str]
    equity_curves: dict[str, list[EquityPoint]]


class ReportRequest(BaseModel):
    """Request body for the RAG report endpoint."""

    ticker: str = "0700.HK"
    start: str = "2022-01-01"
    end: str = "2024-12-31"
    query: str | None = None
    top_k: int = 5
    news_query: str | None = None
    max_news: int = 10


class ReportChunk(BaseModel):
    id: str
    source_type: str
    date: str | None = None
    text: str
    rerank_score: float | None = None


class CitationSummary(BaseModel):
    coverage: float
    orphan_rate: float
    total_citations: int
    supported: int


class UsageSummary(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


class ReportResponse(BaseModel):
    """Response body for the RAG report endpoint."""

    ticker: str
    company_name: str | None = None
    currency: str | None = None
    start: str
    end: str
    query: str
    report: str
    chunks: list[ReportChunk]
    citation: CitationSummary
    usage: UsageSummary
