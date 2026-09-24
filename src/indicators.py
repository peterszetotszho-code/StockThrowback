"""Technical indicator calculations on OHLCV DataFrames.

All functions return a new DataFrame (the input is never mutated) with the
indicator columns appended. They assume a DatetimeIndex and at least a
``Close`` column; Bollinger Bands also only need ``Close``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Trend-state labels. Kept in English (project-wide language rule); the
# semantic meaning is bullish / bearish / ranging.
BULLISH = "bullish"
BEARISH = "bearish"
RANGING = "ranging"


def add_moving_averages(
    df: pd.DataFrame, windows: Sequence[int] = (5, 20, 60)
) -> pd.DataFrame:
    """Append simple moving average columns ``MA{window}`` for each window.

    Args:
        df: OHLCV DataFrame with a ``Close`` column.
        windows: Rolling window sizes.

    Returns:
        A copy of ``df`` with one MA column per window.
    """
    out = df.copy()
    for window in windows:
        out[f"MA{window}"] = out["Close"].rolling(window=window).mean()
    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Append MACD line, signal line, and histogram columns.

    MACD = EMA(fast) - EMA(slow); signal = EMA(MACD); histogram = MACD - signal.

    Args:
        df: OHLCV DataFrame with a ``Close`` column.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal EMA span.

    Returns:
        A copy of ``df`` with columns ``MACD``, ``MACD_Signal``, ``MACD_Hist``.
    """
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_Signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    return out


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Append Wilder's Relative Strength Index as ``RSI``.

    Args:
        df: OHLCV DataFrame with a ``Close`` column.
        period: Lookback period for the smoothed average gains/losses.

    Returns:
        A copy of ``df`` with an ``RSI`` column in [0, 100]; NaN where undefined.
    """
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    return out


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Append Bollinger Bands (upper, middle, lower).

    Args:
        df: OHLCV DataFrame with a ``Close`` column.
        period: Rolling window size for the moving average / standard deviation.
        num_std: Number of standard deviations for the outer bands.

    Returns:
        A copy of ``df`` with ``BB_Upper``, ``BB_Middle``, ``BB_Lower``.
    """
    out = df.copy()
    middle = out["Close"].rolling(window=period).mean()
    std = out["Close"].rolling(window=period).std(ddof=0)
    out["BB_Middle"] = middle
    out["BB_Upper"] = middle + num_std * std
    out["BB_Lower"] = middle - num_std * std
    return out


def define_trend_state(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row as bullish, bearish, or ranging.

    Rule: MA5 > MA20 and MACD > 0 -> bullish;
          MA5 < MA20 and MACD < 0 -> bearish;
          otherwise -> ranging.

    Required columns are computed automatically if absent.

    Args:
        df: OHLCV DataFrame with a ``Close`` column.

    Returns:
        A copy of ``df`` with a ``Trend_State`` column holding one of
        ``bullish`` / ``bearish`` / ``ranging``.
    """
    out = df.copy()
    if "MA5" not in out or "MA20" not in out:
        out = add_moving_averages(out, windows=(5, 20))
    if "MACD" not in out:
        out = add_macd(out)
    conditions = [
        (out["MA5"] > out["MA20"]) & (out["MACD"] > 0),
        (out["MA5"] < out["MA20"]) & (out["MACD"] < 0),
    ]
    choices = [BULLISH, BEARISH]
    out["Trend_State"] = np.select(conditions, choices, default=RANGING)
    return out


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append MA, MACD, RSI, Bollinger Bands, and the trend state in one pass."""
    out = add_moving_averages(df)
    out = add_macd(out)
    out = add_rsi(out)
    out = add_bollinger_bands(out)
    out = define_trend_state(out)
    return out
