"""FastAPI application exposing the backtest pipeline as JSON."""

from __future__ import annotations

import math

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..backtest import run_backtests
from ..data_loader import fetch_stock_data
from ..evaluate import analyze_failures, compare_strategies
from ..indicators import add_all_indicators
from ..strategy import MACDStrategy, MAStrategy, CompositeStrategy
from .schemas import BacktestRequest, BacktestResponse

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
    df = fetch_stock_data(req.ticker, req.start, req.end)
    indicators = add_all_indicators(df)

    strategies = {name: STRATEGIES[name] for name in req.strategies if name in STRATEGIES}
    results = run_backtests(df, strategies, cash=req.cash, commission=req.commission)
    comparison_df = compare_strategies(results)

    return BacktestResponse(
        ticker=req.ticker,
        start=req.start,
        end=req.end,
        candles=_build_candles(indicators),
        comparison=_build_comparison(comparison_df),
        failures={name: analyze_failures(stats) for name, stats in results.items()},
        equity_curves={name: _build_equity(stats) for name, stats in results.items()},
    )


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
