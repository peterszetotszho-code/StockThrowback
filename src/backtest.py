"""Backtest execution on top of ``backtesting.py``."""

from __future__ import annotations

import logging
from typing import Type

import pandas as pd
from backtesting import Backtest, Strategy

logger = logging.getLogger(__name__)

DEFAULT_CASH = 100_000
DEFAULT_COMMISSION = 0.001
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _validate(df: pd.DataFrame) -> None:
    """Raise if the DataFrame is unusable for backtesting."""
    if df is None or df.empty:
        raise ValueError("Cannot backtest on an empty DataFrame.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")


def run_backtest(
    df: pd.DataFrame,
    strategy_class: Type[Strategy],
    cash: float = DEFAULT_CASH,
    commission: float = DEFAULT_COMMISSION,
) -> pd.Series:
    """Run a single backtest and return the full stats object.

    Args:
        df: OHLCV DataFrame (DatetimeIndex, Open/High/Low/Close/Volume).
        strategy_class: A ``backtesting.Strategy`` subclass.
        cash: Initial cash.
        commission: Commission rate applied per trade (0.001 = 0.1%).

    Returns:
        A pandas Series of backtest statistics (Sharpe ratio, max drawdown,
        win rate, profit factor, trade count, etc.).
    """
    _validate(df)
    logger.info(
        "Running %s on %d bars (cash=%s, commission=%s)",
        strategy_class.__name__,
        len(df),
        cash,
        commission,
    )
    bt = Backtest(df, strategy_class, cash=cash, commission=commission, exclusive_orders=True)
    return bt.run()


def run_backtests(
    df: pd.DataFrame,
    strategies: dict[str, Type[Strategy]],
    cash: float = DEFAULT_CASH,
    commission: float = DEFAULT_COMMISSION,
) -> dict[str, pd.Series]:
    """Run several strategies on the same data and return name -> stats.

    Args:
        df: OHLCV DataFrame.
        strategies: Mapping of display name -> strategy class.
        cash: Initial cash.
        commission: Commission rate per trade.

    Returns:
        Mapping of display name -> stats Series, in the given order.
    """
    results: dict[str, pd.Series] = {}
    for name, strategy_class in strategies.items():
        logger.info("Backtesting strategy '%s' ...", name)
        results[name] = run_backtest(df, strategy_class, cash=cash, commission=commission)
    return results
