"""Tests for the backtest runner and parameter optimizer."""

from src.backtest import optimize_parameters
from src.strategy import MAStrategy


def test_optimize_parameters(ohlcv):
    best = optimize_parameters(ohlcv, MAStrategy, fast=[5, 10], slow=[20, 30], maximize="Sharpe Ratio")
    assert "Sharpe Ratio" in best.index
    # backtesting.py sets the winning parameter values on the strategy class.
    assert MAStrategy.fast in (5, 10)
    assert MAStrategy.slow in (20, 30)
