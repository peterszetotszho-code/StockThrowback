"""Tests for the FastAPI backend (no network; data fetch is monkeypatched)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def _synthetic_df() -> pd.DataFrame:
    idx = pd.date_range("2022-01-03", periods=120, freq="B")
    close = 100 + np.cumsum(np.sin(np.linspace(0, 20, 120))) * 0.5
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 100_000,
        },
        index=idx,
    )


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_backtest(monkeypatch):
    from src.api import main as api_main

    monkeypatch.setattr(api_main, "fetch_stock_data", lambda ticker, start, end: _synthetic_df())
    response = client.post(
        "/api/backtest",
        json={"ticker": "TEST.HK", "start": "2022-01-01", "end": "2022-06-01"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TEST.HK"
    assert len(data["candles"]) == 120
    assert len(data["comparison"]) == 3
    assert "MA" in data["equity_curves"]
    assert data["candles"][0]["close"] > 0
