"""Performance comparison and honest failure analysis."""

from __future__ import annotations

import logging

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
        worst_date = equity["DrawdownPct"].idxmin()
        worst_value = equity["DrawdownPct"].min()
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
