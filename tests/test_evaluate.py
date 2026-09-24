"""Tests for evaluation helpers."""

import pandas as pd

from src import evaluate
from src.strategy import MAStrategy


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


def test_walk_forward_shape(ohlcv):
    result = evaluate.walk_forward(ohlcv, MAStrategy, n_splits=5)
    assert len(result) == 5
    assert list(result.columns) == ["start", "end", "return_pct", "sharpe", "max_drawdown_pct", "trades"]


def test_split_evaluate_shape(ohlcv):
    result = evaluate.split_evaluate(ohlcv, MAStrategy, train_ratio=0.7)
    assert list(result.index) == ["in_sample", "out_of_sample"]
    assert "sharpe" in result.columns


def test_monte_carlo(ohlcv):
    result = evaluate.monte_carlo(ohlcv, MAStrategy, n_runs=5, seed=0)
    assert "mean_return_pct" in result.index
    assert 0 <= result["win_rate_pct"] <= 100
