"""Result validators — the project's honest-evaluation gate."""

from __future__ import annotations

import pandas as pd

from .tool import ValidationError


def validate_non_empty_dataframe(result: object, required_columns: list[str]) -> None:
    """Require a non-empty DataFrame containing every required column."""
    if not isinstance(result, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(result).__name__}")
    if result.empty:
        raise ValidationError("DataFrame is empty")
    missing = [c for c in required_columns if c not in result.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")


def validate_backtest_result(stats: object) -> None:
    """Require backtesting.py run() output to carry the core stats keys."""
    if not isinstance(stats, pd.Series):
        raise ValidationError(f"Expected pd.Series of stats, got {type(stats).__name__}")
    required = ["Return [%]", "Sharpe Ratio", "Max. Drawdown [%]", "Win Rate [%]"]
    missing = [key for key in required if key not in stats.index]
    if missing:
        raise ValidationError(f"Backtest stats missing keys: {missing}")
