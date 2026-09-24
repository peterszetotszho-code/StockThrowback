"""Tests for data_loader using a mocked downloader."""

import numpy as np
import pandas as pd
import pytest

from src import data_loader


def _sample_frame() -> pd.DataFrame:
    idx = pd.date_range("2022-01-03", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "High": [2.0, 3.0, 4.0, 5.0, 6.0],
            "Low": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Volume": [100, 200, 300, 400, 500],
        },
        index=idx,
    )


def test_fetch_uses_cache(monkeypatch, tmp_path):
    yf = pytest.importorskip("yfinance")
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        return _sample_frame()

    monkeypatch.setattr(yf, "download", fake_download)
    df1 = data_loader.fetch_stock_data("TEST.HK", "2022-01-01", "2022-01-07", cache_dir=tmp_path)
    df2 = data_loader.fetch_stock_data("TEST.HK", "2022-01-01", "2022-01-07", cache_dir=tmp_path)

    assert calls["n"] == 1
    # `freq` is metadata that a parquet round-trip does not preserve; the
    # values are what matter for stock data (which trades on irregular days).
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)
    assert list(df1.columns) == data_loader.PRICE_COLUMNS


def test_fetch_empty_raises(monkeypatch, tmp_path):
    yf = pytest.importorskip("yfinance")
    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError):
        data_loader.fetch_stock_data("EMPTY.HK", "2022-01-01", "2022-01-07", cache_dir=tmp_path)


def test_normalize_multiindex_columns():
    idx = pd.date_range("2022-01-03", periods=3, freq="D")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["X"]])
    df = pd.DataFrame(np.arange(15).reshape(3, 5), index=idx, columns=cols)
    out = data_loader._normalize_columns(df)
    assert list(out.columns) == data_loader.PRICE_COLUMNS


def test_get_company_name(monkeypatch):
    import yfinance as yf

    class FakeTicker:
        info = {"shortName": "Tencent Holdings"}

    monkeypatch.setattr(yf, "Ticker", lambda ticker: FakeTicker())
    assert data_loader.get_company_name("0700.HK") == "Tencent Holdings"


def test_get_company_name_returns_none_on_error(monkeypatch):
    import yfinance as yf

    def boom(ticker):
        raise RuntimeError("network down")

    monkeypatch.setattr(yf, "Ticker", boom)
    assert data_loader.get_company_name("0700.HK") is None


def test_clean_fills_volume_gap():
    idx = pd.date_range("2022-01-03", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0],
            "High": [2.0, 3.0, 4.0, 5.0],
            "Low": [0.0, 1.0, 2.0, 3.0],
            "Close": [1.5, 2.5, 3.5, 4.5],
            "Volume": [100, None, 300, 400],
        },
        index=idx,
    )
    out = data_loader._clean(df)
    assert out["Volume"].iloc[1] == 0
    assert out["Volume"].dtype == "int64"
    assert out.index.name == "Date"
