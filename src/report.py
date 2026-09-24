"""Markdown report generation for backtest results."""

from __future__ import annotations

import pandas as pd


def generate_report(
    ticker: str,
    comparison_df: pd.DataFrame,
    failure_analysis: str | dict[str, str],
    start: str | None = None,
    end: str | None = None,
) -> str:
    """Render a Markdown analysis report.

    Args:
        ticker: The ticker symbol under review.
        comparison_df: Strategy comparison table (from ``compare_strategies``).
        failure_analysis: A single failure-analysis string, or a per-strategy
            mapping of name -> analysis text.
        start: Optional start date for the data-range header.
        end: Optional end date for the data-range header.

    Returns:
        A Markdown document as a string.
    """
    lines = [
        f"# Stock Trend Analysis Report - {ticker}",
        "",
        "## Data Range",
        "",
        _format_range(start, end),
        "",
        "## Strategy Comparison",
        "",
        comparison_df.round(2).to_markdown(index=True),
        "",
        "## Failure Analysis",
        "",
    ]

    if isinstance(failure_analysis, dict):
        for name, text in failure_analysis.items():
            lines += [f"### {name}", "", text, ""]
    else:
        lines += [failure_analysis, ""]

    lines += ["## Conclusion", "", _build_conclusion(comparison_df), ""]
    return "\n".join(lines)


def _format_range(start: str | None, end: str | None) -> str:
    """Format the data-range line from optional start/end dates."""
    if start and end:
        return f"From {start} to {end}."
    if start or end:
        return f"Partial range: {start or '?'} to {end or '?'}."
    return "Data range not provided."


def _build_conclusion(comparison_df: pd.DataFrame) -> str:
    """Summarize the comparison table with an honest, caveated conclusion."""
    if comparison_df.empty:
        return "No strategies were evaluated, so no conclusion can be drawn."

    if "Sharpe Ratio" in comparison_df.columns:
        best = comparison_df["Sharpe Ratio"].idxmax()
        best_sharpe = comparison_df.loc[best, "Sharpe Ratio"]
    else:
        best = comparison_df.index[0]
        best_sharpe = float("nan")

    return (
        f"Across the evaluated strategies, '{best}' posted the highest Sharpe "
        f"ratio ({best_sharpe:.2f}). Past performance is not indicative of "
        "future results; treat these figures as an in-sample engineering "
        "exercise, not a trading recommendation."
    )
