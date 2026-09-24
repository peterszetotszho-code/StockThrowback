"""Tests for report generation."""

import pandas as pd
import pytest

from src import evaluate, report


def _stats(**overrides) -> pd.Series:
    base = {
        "Return [%]": 12.5,
        "Sharpe Ratio": 1.2,
        "Max. Drawdown [%]": -8.0,
        "Win Rate [%]": 55.0,
        "Profit Factor": 1.4,
        "# Trades": 10,
    }
    base.update(overrides)
    return pd.Series(base)


def test_generate_report_contains_required_sections():
    pytest.importorskip("tabulate")
    comparison = evaluate.compare_strategies({"MA": _stats(), "MACD": _stats()})
    md = report.generate_report(
        "0700.HK", comparison, "No trades executed.", start="2022-01-01", end="2024-12-31"
    )
    for section in [
        "# Stock Trend Analysis Report",
        "## Data Range",
        "## Strategy Comparison",
        "## Failure Analysis",
        "## Conclusion",
    ]:
        assert section in md
    assert "2022-01-01" in md


def test_generate_report_dict_failures():
    pytest.importorskip("tabulate")
    comparison = evaluate.compare_strategies({"MA": _stats()})
    md = report.generate_report("AAPL", comparison, {"MA": "some analysis"})
    assert "### MA" in md
    assert "some analysis" in md
