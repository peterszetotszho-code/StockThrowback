"""FastAPI application exposing the backtest pipeline as JSON."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..backtest import run_backtests
from ..data_loader import fetch_stock_data, get_ticker_info
from ..evaluate import analyze_failures, compare_strategies
from ..indicators import add_all_indicators
from ..rag.pipeline import run_report_pipeline
from ..strategy import MACDStrategy, MAStrategy, CompositeStrategy
from .schemas import BacktestRequest, BacktestResponse, ReportRequest, ReportResponse

STRATEGIES = {"MA": MAStrategy, "MACD": MACDStrategy, "Composite": CompositeStrategy}

app = FastAPI(title="stock-trend-lab API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/api/backtest", response_model=BacktestResponse)
def run_analysis(req: BacktestRequest) -> BacktestResponse:
    """Fetch data, run the selected strategies, and return chart + metrics JSON."""
    if req.start > req.end:
        raise HTTPException(status_code=422, detail="Start date must be on or before the end date.")
    try:
        df = fetch_stock_data(req.ticker, req.start, req.end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    indicators = add_all_indicators(df)

    strategies = {name: STRATEGIES[name] for name in req.strategies if name in STRATEGIES}
    results = run_backtests(df, strategies, cash=req.cash, commission=req.commission)
    comparison_df = compare_strategies(results)

    info = get_ticker_info(req.ticker)
    return BacktestResponse(
        ticker=req.ticker,
        company_name=info["company_name"],
        currency=info["currency"],
        start=req.start,
        end=req.end,
        candles=_build_candles(indicators),
        comparison=_build_comparison(comparison_df),
        failures={name: analyze_failures(stats) for name, stats in results.items()},
        equity_curves={name: _build_equity(stats) for name, stats in results.items()},
    )


@app.post("/api/report", response_model=ReportResponse)
def generate_report(req: ReportRequest) -> ReportResponse:
    """Run the backtest + RAG + citation + usage pipeline and return the report."""
    if req.start > req.end:
        raise HTTPException(status_code=422, detail="Start date must be on or before the end date.")
    try:
        result = run_report_pipeline(
            req.ticker,
            req.start,
            req.end,
            query=req.query,
            top_k=req.top_k,
            news_query=req.news_query,
            max_news=req.max_news,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    info = get_ticker_info(req.ticker)
    result["company_name"] = info["company_name"]
    result["currency"] = info["currency"]
    return ReportResponse(**result)


def _to_seconds(ts: pd.Timestamp) -> int:
    return int(ts.timestamp())


def _optional(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    # Guard against inf/NaN (e.g. Profit Factor with no losing trades).
    return result if math.isfinite(result) else None


def _build_candles(df: pd.DataFrame) -> list[dict]:
    candles: list[dict] = []
    for idx, row in df.iterrows():
        candles.append(
            {
                "time": _to_seconds(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "ma5": _optional(row.get("MA5")),
                "ma20": _optional(row.get("MA20")),
                "ma60": _optional(row.get("MA60")),
                "rsi": _optional(row.get("RSI")),
                "trend": row.get("Trend_State", "ranging"),
            }
        )
    return candles


def _build_comparison(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for name, row in df.iterrows():
        rows.append(
            {
                "strategy": name,
                "return_pct": _optional(row.get("Return [%]")),
                "sharpe": _optional(row.get("Sharpe Ratio")),
                "max_drawdown_pct": _optional(row.get("Max. Drawdown [%]")),
                "win_rate_pct": _optional(row.get("Win Rate [%]")),
                "profit_factor": _optional(row.get("Profit Factor")),
                "trades": int(row["Trades"]) if pd.notna(row.get("Trades")) else 0,
            }
        )
    return rows


def _build_equity(stats: pd.Series) -> list[dict]:
    equity = getattr(stats, "_equity_curve", None)
    if equity is None or equity.empty:
        return []
    points: list[dict] = []
    for idx, row in equity.iterrows():
        points.append(
            {
                "time": _to_seconds(idx),
                "equity": float(row["Equity"]),
                # backtesting.py stores drawdown as a positive fraction.
                "drawdown": -float(row["DrawdownPct"]) * 100.0,
            }
        )
    return points


# Serve the built React frontend when it is present (e.g. the Docker image).
# In local development Vite serves the frontend on :5173, so this is a no-op
# unless ``frontend/dist`` exists on disk.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
