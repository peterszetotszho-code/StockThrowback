"""Performance comparison and honest failure analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get(stats: pd.Series, key: str):
    """Return a stat value or NaN when the key is absent."""
    return stats.get(key, float("nan"))


def compare_strategies(results: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a side-by-side comparison table from per-strategy stats.

    Args:
        results: Mapping of strategy name -> backtesting stats Series.

    Returns:
        DataFrame indexed by strategy name with columns Return [%],
        Sharpe Ratio, Max. Drawdown [%], Win Rate [%], Profit Factor, Trades.
    """
    rows = []
    for name, stats in results.items():
        rows.append(
            {
                "Return [%]": _get(stats, "Return [%]"),
                "Sharpe Ratio": _get(stats, "Sharpe Ratio"),
                "Max. Drawdown [%]": _get(stats, "Max. Drawdown [%]"),
                "Win Rate [%]": _get(stats, "Win Rate [%]"),
                "Profit Factor": _get(stats, "Profit Factor"),
                "Trades": _get(stats, "# Trades"),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(list(results), name="Strategy"))


def analyze_failures(stats: pd.Series) -> str:
    """Produce a plain-language failure analysis for a single backtest.

    This is intentionally heuristic and honest: it reports the observed losing
    trades, the worst drawdown window, and a qualitative caveat, without
    fabricating precision the data does not support.

    Args:
        stats: A backtesting stats Series.

    Returns:
        A string describing where and why the strategy struggled.
    """
    trades = int(stats.get("# Trades", 0) or 0)
    if trades == 0:
        return (
            "No trades were executed during the period, so there is no track "
            "record to evaluate - the strategy never produced a signal."
        )

    win_rate = _get(stats, "Win Rate [%]")
    wins = int(round(trades * win_rate / 100.0)) if pd.notna(win_rate) else 0
    losses = trades - wins
    profit_factor = _get(stats, "Profit Factor")
    max_dd = _get(stats, "Max. Drawdown [%]")

    lines = [
        f"Executed {trades} trades with a win rate of {win_rate:.1f}% "
        f"({wins} wins, {losses} losses)."
    ]

    if pd.notna(max_dd):
        lines.append(f"Maximum drawdown was {max_dd:.1f}%.")

    # Locate the worst drawdown window from the equity curve when available.
    equity = getattr(stats, "_equity_curve", None)
    if equity is not None and not equity.empty and "DrawdownPct" in equity.columns:
        drawdown = equity["DrawdownPct"]
        # backtesting.py stores drawdown as a positive fraction (0 = none).
        # Locate the largest magnitude, convert to percent, and report it as
        # negative to match the "Max. Drawdown [%]" stat convention.
        worst_date = drawdown.abs().idxmax()
        worst_value = -abs(drawdown.loc[worst_date]) * 100.0
        lines.append(
            f"The deepest equity dip of {worst_value:.1f}% was reached on "
            f"{worst_date:%Y-%m-%d}."
        )

    if losses > wins:
        lines.append(
            "More losing than winning trades suggests the strategy was whipsawed "
            "by choppy or ranging conditions, generating false signals that a "
            "trend filter would likely reduce."
        )
    elif pd.notna(profit_factor) and profit_factor < 1.0:
        lines.append(
            "Profit factor below 1.0 means average losses exceeded average wins, "
            "so the edge did not survive transaction costs."
        )
    else:
        lines.append(
            "Overall statistics look positive, but remember this is a single "
            "in-sample backtest and says nothing about out-of-sample robustness."
        )

    return " ".join(lines)


def walk_forward(
    df: pd.DataFrame,
    strategy_class,
    n_splits: int = 5,
    cash: float = 100_000,
    commission: float = 0.001,
) -> pd.DataFrame:
    """Evaluate a strategy on contiguous, non-overlapping windows.

    Fixed-parameter strategies have no "training", so walk-forward here means:
    split the series into ``n_splits`` equal, contiguous windows and evaluate
    the strategy independently on each. A strategy that stays positive across
    later windows is more robust than one whose gains come from a single lucky
    window.

    Args:
        df: OHLCV DataFrame.
        strategy_class: Strategy subclass.
        n_splits: Number of contiguous windows.
        cash: Initial cash.
        commission: Commission rate.

    Returns:
        DataFrame indexed by window (1..n) with start, end, return_pct,
        sharpe, max_drawdown_pct, trades.
    """
    from .backtest import run_backtest

    rows: list[dict] = []
    for idx in np.array_split(np.arange(len(df)), n_splits):
        if idx.size == 0:
            continue
        window = df.iloc[idx]
        stats = run_backtest(window, strategy_class, cash=cash, commission=commission)
        rows.append(
            {
                "start": window.index[0].strftime("%Y-%m-%d"),
                "end": window.index[-1].strftime("%Y-%m-%d"),
                "return_pct": _get(stats, "Return [%]"),
                "sharpe": _get(stats, "Sharpe Ratio"),
                "max_drawdown_pct": _get(stats, "Max. Drawdown [%]"),
                "trades": int(_get(stats, "# Trades") or 0),
            }
        )
    result = pd.DataFrame(rows)
    result.index = pd.Index(range(1, len(rows) + 1), name="window")
    return result


def split_evaluate(
    df: pd.DataFrame,
    strategy_class,
    train_ratio: float = 0.7,
    cash: float = 100_000,
    commission: float = 0.001,
) -> pd.DataFrame:
    """Compare in-sample vs out-of-sample performance on a single split.

    The first ``train_ratio`` of the data is in-sample; the remainder is
    out-of-sample. Degradation between the two is a warning sign of overfitting
    (for parameterized strategies) or regime dependence (for fixed rules).

    Args:
        df: OHLCV DataFrame.
        strategy_class: Strategy subclass.
        train_ratio: Fraction of rows used as the in-sample window.
        cash: Initial cash.
        commission: Commission rate.

    Returns:
        DataFrame indexed by ["in_sample", "out_of_sample"] with columns
        return_pct, sharpe, max_drawdown_pct, trades.
    """
    from .backtest import run_backtest

    split = int(len(df) * train_ratio)
    windows = (("in_sample", df.iloc[:split]), ("out_of_sample", df.iloc[split:]))
    rows: dict[str, dict] = {}
    for label, window in windows:
        if window.empty:
            continue
        stats = run_backtest(window, strategy_class, cash=cash, commission=commission)
        rows[label] = {
            "return_pct": _get(stats, "Return [%]"),
            "sharpe": _get(stats, "Sharpe Ratio"),
            "max_drawdown_pct": _get(stats, "Max. Drawdown [%]"),
            "trades": int(_get(stats, "# Trades") or 0),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def monte_carlo(
    df: pd.DataFrame,
    strategy_class,
    n_runs: int = 20,
    window_frac: float = 0.5,
    seed: int = 0,
    cash: float = 100_000,
    commission: float = 0.001,
) -> pd.Series:
    """Estimate return dispersion by backtesting on random contiguous windows.

    Samples ``n_runs`` random contiguous windows of length ``window_frac`` and
    reports the distribution of returns, a cheap proxy for robustness to the
    chosen time period.

    Args:
        df: OHLCV DataFrame.
        strategy_class: Strategy subclass.
        n_runs: Number of random windows.
        window_frac: Fraction of the series used per window.
        seed: RNG seed for reproducibility.
        cash: Initial cash.
        commission: Commission rate.

    Returns:
        A Series of summary stats (mean/median/min/max return, win rate).
    """
    from .backtest import run_backtest

    rng = np.random.default_rng(seed)
    n = len(df)
    window_len = max(int(n * window_frac), 2)
    returns: list[float] = []
    for _ in range(n_runs):
        start = int(rng.integers(0, n - window_len + 1))
        window = df.iloc[start : start + window_len]
        stats = run_backtest(window, strategy_class, cash=cash, commission=commission)
        value = _get(stats, "Return [%]")
        if pd.notna(value):
            returns.append(float(value))
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return pd.Series(
            {"mean_return_pct": float("nan"), "median_return_pct": float("nan"),
             "min_return_pct": float("nan"), "max_return_pct": float("nan"),
             "win_rate_pct": float("nan")}
        )
    return pd.Series(
        {
            "mean_return_pct": float(arr.mean()),
            "median_return_pct": float(np.median(arr)),
            "min_return_pct": float(arr.min()),
            "max_return_pct": float(arr.max()),
            "win_rate_pct": float((arr > 0).mean() * 100),
        }
    )
