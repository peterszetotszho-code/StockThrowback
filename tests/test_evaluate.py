"""Tests for evaluation helpers."""

import pandas as pd

from src import evaluate


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


def test_compare_strategies_shape_and_columns():
    results = {"A": _stats(), "B": _stats(**{"Return [%]": 5.0, "# Trades": 3})}
    df = evaluate.compare_strategies(results)
    assert list(df.index) == ["A", "B"]
    assert {"Return [%]", "Sharpe Ratio", "Max. Drawdown [%]", "Win Rate [%]", "Trades"} <= set(df.columns)
    assert df.loc["B", "Trades"] == 3


def test_analyze_failures_no_trades():
    msg = evaluate.analyze_failures(_stats(**{"# Trades": 0}))
    assert "No trades" in msg


def test_analyze_failures_losing():
    msg = evaluate.analyze_failures(_stats(**{"# Trades": 10, "Win Rate [%]": 30.0}))
    assert "whipsawed" in msg
