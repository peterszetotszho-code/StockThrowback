"""Concrete tool implementations wrapping the core services."""

from __future__ import annotations

from ..backtest import run_backtest as _run_backtest
from ..data_loader import fetch_stock_data
from ..evaluate import analyze_failures as _analyze_failures
from ..strategy import MACDStrategy, MAStrategy, CompositeStrategy
from .tool import Tool, ToolRegistry, ValidationError
from .validation import validate_backtest_result, validate_non_empty_dataframe

_STRATEGIES = {"MA": MAStrategy, "MACD": MACDStrategy, "Composite": CompositeStrategy}
_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _run_backtest_tool(df, strategy: str = "MA"):
    """Backtest a named strategy on an OHLCV DataFrame."""
    strategy_class = _STRATEGIES.get(strategy)
    if strategy_class is None:
        raise ValidationError(f"Unknown strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}")
    return _run_backtest(df, strategy_class)


def build_default_registry() -> ToolRegistry:
    """Return a registry with the core backtest-analysis tools."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fetch_market_data",
            description="Download OHLCV market data for a ticker over an inclusive date range.",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Yahoo Finance ticker, e.g. 0700.HK"},
                    "start": {"type": "string", "description": "Start date, YYYY-MM-DD"},
                    "end": {"type": "string", "description": "End date, YYYY-MM-DD"},
                },
                "required": ["ticker", "start", "end"],
            },
            fn=fetch_stock_data,
            validator=lambda df: validate_non_empty_dataframe(df, _OHLCV_COLUMNS),
        )
    )
    registry.register(
        Tool(
            name="run_backtest",
            description="Backtest a named strategy (MA, MACD, or Composite) on OHLCV data.",
            parameters={
                "type": "object",
                "properties": {
                    "df": {"type": "object", "description": "OHLCV DataFrame from a prior step"},
                    "strategy": {"type": "string", "enum": sorted(_STRATEGIES)},
                },
                "required": ["df", "strategy"],
            },
            fn=_run_backtest_tool,
            validator=validate_backtest_result,
        )
    )
    registry.register(
        Tool(
            name="analyze_failures",
            description="Produce a plain-language failure analysis from backtest stats.",
            parameters={
                "type": "object",
                "properties": {
                    "stats": {"type": "object", "description": "Backtest stats Series from a prior step"}
                },
                "required": ["stats"],
            },
            fn=_analyze_failures,
        )
    )
    return registry
