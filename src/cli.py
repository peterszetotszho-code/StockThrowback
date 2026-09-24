"""Command-line entry point that runs the full analysis pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .backtest import run_backtests
from .data_loader import fetch_stock_data
from .evaluate import analyze_failures, compare_strategies
from .logging_config import configure_logging
from .report import generate_report
from .strategy import MACDStrategy, MAStrategy, CompositeStrategy

logger = logging.getLogger(__name__)

STRATEGIES = {
    "MA": MAStrategy,
    "MACD": MACDStrategy,
    "Composite": CompositeStrategy,
}


def main(argv: list[str] | None = None) -> int:
    """Run the full fetch -> backtest -> evaluate -> report pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="stock-trend-lab backtest CLI")
    parser.add_argument("--ticker", default="0700.HK", help="Yahoo Finance ticker")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Optional path to save the Markdown report")
    args = parser.parse_args(argv)

    configure_logging()

    df = fetch_stock_data(args.ticker, args.start, args.end)

    results = run_backtests(df, STRATEGIES)
    comparison = compare_strategies(results)
    failures = {name: analyze_failures(stats) for name, stats in results.items()}

    report_md = generate_report(args.ticker, comparison, failures, start=args.start, end=args.end)

    print(comparison.round(2).to_string())
    print("\n" + report_md)

    if args.output:
        Path(args.output).write_text(report_md, encoding="utf-8")
        logger.info("Report saved to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
