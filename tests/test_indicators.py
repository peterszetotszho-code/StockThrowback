"""Tests for technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from src import indicators


def _frame(close) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D")
    return pd.DataFrame({"Close": close}, index=idx)


def test_add_moving_averages_values():
    df = _frame([1.0, 2.0, 3.0, 4.0, 5.0])
    out = indicators.add_moving_averages(df, windows=(3,))
    assert "MA3" in out
    assert out["MA3"].iloc[:2].isna().all()
    assert out["MA3"].iloc[2] == pytest.approx(2.0)
    assert out["MA3"].iloc[4] == pytest.approx(4.0)


def test_add_macd_identity():
    df = _frame(np.linspace(1, 50, 50))
    out = indicators.add_macd(df)
    assert not out["MACD"].isna().any()
    assert (out["MACD_Hist"] == out["MACD"] - out["MACD_Signal"]).all()


def test_add_rsi_extremes():
    up = _frame(np.arange(1, 31, dtype=float))
    out_up = indicators.add_rsi(up, period=14)
    assert out_up["RSI"].iloc[:13].isna().all()
    assert (out_up["RSI"].iloc[15:] > 99).all()

    down = _frame(np.arange(30, 0, -1, dtype=float))
    out_down = indicators.add_rsi(down, period=14)
    assert (out_down["RSI"].iloc[15:] < 1).all()


def test_add_bollinger_bands_geometry():
    df = _frame(np.linspace(1, 100, 100))
    out = indicators.add_bollinger_bands(df, period=20, num_std=2.0)
    middle = df["Close"].rolling(20).mean()
    pd.testing.assert_series_equal(out["BB_Middle"], middle, check_names=False)
    width = out["BB_Upper"] - out["BB_Lower"]
    expected = 2 * 2.0 * df["Close"].rolling(20).std(ddof=0)
    assert np.allclose(width.iloc[20:].to_numpy(), expected.iloc[20:].to_numpy())


def test_define_trend_state():
    up = _frame(np.linspace(1, 100, 100))
    assert (indicators.define_trend_state(up)["Trend_State"].iloc[-10:] == "bullish").all()

    down = _frame(np.linspace(100, 1, 100))
    assert (indicators.define_trend_state(down)["Trend_State"].iloc[-10:] == "bearish").all()

    flat = _frame([1.0] * 100)
    assert (indicators.define_trend_state(flat)["Trend_State"] == "ranging").all()


def test_trend_state_values_are_valid():
    rng = np.random.default_rng(0)
    df = _frame(rng.normal(100, 2, 200).cumsum())
    out = indicators.define_trend_state(df)
    assert set(out["Trend_State"].unique()) <= {"bullish", "bearish", "ranging"}
