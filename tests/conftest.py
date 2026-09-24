"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """A deterministic, realistic-looking OHLCV frame (150 business days)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=150, freq="B")
    close = np.abs(100 + np.cumsum(rng.normal(0, 1, len(idx)))) + 50
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
