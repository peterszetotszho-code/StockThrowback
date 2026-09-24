"""Tests for the FastAPI backend (no network; data fetch is monkeypatched)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.embeddings import HashEmbedder

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


def _no_company_name(monkeypatch):
    """Prevent real yfinance name lookups during tests."""
    from src.api import main as api_main

    monkeypatch.setattr(
        api_main, "get_ticker_info", lambda ticker: {"company_name": None, "currency": None}
    )


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_backtest(monkeypatch):
    from src.api import main as api_main

    monkeypatch.setattr(api_main, "fetch_stock_data", lambda ticker, start, end: _synthetic_df())
    _no_company_name(monkeypatch)
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


def test_report(monkeypatch):
    from src.rag import pipeline

    monkeypatch.setattr(pipeline, "fetch_stock_data", lambda ticker, start, end: _synthetic_df())
    monkeypatch.setattr(pipeline, "_auto_llm", lambda: None)
    monkeypatch.setattr(pipeline, "get_embedder", lambda: HashEmbedder(dim=128))
    _no_company_name(monkeypatch)
    response = client.post(
        "/api/report",
        json={"ticker": "TEST.HK", "start": "2022-01-01", "end": "2022-06-01"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TEST.HK"
    assert data["report"].startswith("# Report")
    assert len(data["chunks"]) >= 5
    assert data["citation"]["total_citations"] == len(data["chunks"])
    assert data["usage"]["calls"] >= 3


def test_report_includes_knowledge(monkeypatch):
    from src.rag import pipeline

    monkeypatch.setattr(pipeline, "fetch_stock_data", lambda ticker, start, end: _synthetic_df())
    monkeypatch.setattr(pipeline, "_auto_llm", lambda: None)
    monkeypatch.setattr(pipeline, "get_embedder", lambda: HashEmbedder(dim=128))
    _no_company_name(monkeypatch)
    response = client.post(
        "/api/report",
        json={"ticker": "TEST.HK", "start": "2022-01-01", "end": "2022-06-01"},
    )
    assert response.status_code == 200
    data = response.json()
    assert any(c["source_type"] == "knowledge" for c in data["chunks"])
    assert "Knowledge context" in data["report"]


def test_backtest_invalid_date_range():
    response = client.post(
        "/api/backtest",
        json={"ticker": "TEST.HK", "start": "2023-01-01", "end": "2022-01-01"},
    )
    assert response.status_code == 422
    assert "end date" in response.json()["detail"]


def test_backtest_invalid_ticker(monkeypatch):
    from src.api import main as api_main

    def boom(ticker, start, end):
        raise ValueError(f"No data returned for {ticker} in the given range.")

    monkeypatch.setattr(api_main, "fetch_stock_data", boom)
    response = client.post(
        "/api/backtest",
        json={"ticker": "INVALID", "start": "2022-01-01", "end": "2022-06-01"},
    )
    assert response.status_code == 422
    assert "No data returned" in response.json()["detail"]


def test_auto_llm_returns_none_without_key(monkeypatch):
    from src.rag import pipeline

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert pipeline._auto_llm() is None


def test_auto_llm_uses_deepseek(monkeypatch):
    from src.rag import pipeline

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = pipeline._auto_llm()
    assert llm is not None
    assert llm.model == "deepseek-chat"
    assert llm._base_url == "https://api.deepseek.com"


def test_report_uses_llm(monkeypatch):
    from src.rag import pipeline
    from src.agent.llm import FakeChatModel, LLMResponse

    monkeypatch.setattr(pipeline, "fetch_stock_data", lambda t, s, e: _synthetic_df())
    monkeypatch.setattr(pipeline, "get_embedder", lambda: HashEmbedder(dim=128))
    llm = FakeChatModel([LLMResponse(content="# LLM report\n\nCites [1] and [2].")])
    result = pipeline.run_report_pipeline("TEST.HK", "2022-01-01", "2022-06-01", llm=llm)
    assert result["report"].startswith("# LLM report")
    assert result["citation"]["total_citations"] == 2
    assert result["citation"]["orphan_rate"] == 0.0


def test_report_with_news(monkeypatch):
    from src.rag import pipeline

    monkeypatch.setattr(pipeline, "fetch_stock_data", lambda ticker, start, end: _synthetic_df())
    monkeypatch.setattr(pipeline, "_auto_llm", lambda: None)
    monkeypatch.setattr(pipeline, "get_embedder", lambda: HashEmbedder(dim=128))
    _no_company_name(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "fetch_gdelt_articles",
        lambda q, s, e, max_records=10: [
            {"title": "TEST.HK draws down on news", "url": "http://x", "date": "2022-03-01", "source_id": "g0"}
        ],
    )
    response = client.post(
        "/api/report",
        json={
            "ticker": "TEST.HK",
            "start": "2022-01-01",
            "end": "2022-06-01",
            "news_query": "Tencent",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert any(c["source_type"] == "news" for c in data["chunks"])
    assert "News during worst drawdown" in data["report"]
