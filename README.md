# stock-trend-lab

Systematic historical trend review and strategy backtesting for individual stocks.

> This project does **not** predict future prices. It engineers a reproducible
> pipeline to replay historical trends, backtest technical-indicator strategies,
> and evaluate their performance honestly — including failure analysis.

## Tech Stack

- Python 3.10+
- [`yfinance`](https://github.com/ranaroussi/yfinance) — market data
- `pandas` / `numpy` — data processing
- [`backtesting.py`](https://github.com/kernc/backtesting.py) — event-driven backtesting
- `matplotlib` — visualization
- `pyarrow` — parquet cache
- `pytest` — tests
- Optional (`requirements-optional.txt`): `chromadb`, `rank-bm25`,
  `sentence-transformers`, `openai` — RAG retrieval, reranking, and LLM reports

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest
python -m src.cli --ticker 0700.HK --start 2022-01-01 --end 2024-12-31 --output outputs/reports/0700.HK.md
```

The CLI downloads live data (requires internet). `python -m pytest` runs fully
offline and validates the core modules.

## Project Structure

```
stock-trend-lab/
├── src/
│   ├── data_loader.py      # data fetch + parquet cache + cleaning
│   ├── indicators.py       # MA / MACD / RSI / Bollinger / trend state
│   ├── strategy.py         # MA / MACD / Composite strategies
│   ├── backtest.py         # run + batch backtests
│   ├── evaluate.py         # strategy comparison + failure analysis
│   ├── report.py           # Markdown report generation
│   └── cli.py              # end-to-end entry point
├── tests/                  # pytest suite
├── notebooks/              # exploratory analysis
├── knowledge/              # documents for the RAG knowledge base
├── outputs/                # cached data, vector store, reports (gitignored)
└── requirements*.txt
```

## Evaluation Results

A real, reproducible run (no fabricated numbers):

- Ticker `0700.HK` (Tencent), 2022-01-01 to 2024-12-31 (734 trading days)
- Initial cash $100,000, commission 0.1%, open trades finalized at period end

Reproduce with:

```bash
python -m src.cli --ticker 0700.HK --start 2022-01-01 --end 2024-12-31
```

| Strategy | Return [%] | Sharpe Ratio | Max. Drawdown [%] | Win Rate [%] | Profit Factor | Trades |
|---|---|---|---|---|---|---|
| MA | 30.91 | 0.36 | -28.83 | 22.73 | 1.81 | 22 |
| MACD | 18.69 | 0.22 | -36.82 | 39.13 | 1.40 | 23 |
| Composite | 30.91 | 0.36 | -28.83 | 22.73 | 1.81 | 22 |

## Failure Analysis

Honest findings from the same run:

- **Low win rate, positive profit factor.** MA wins only ~23% of trades yet ends
  profitable, because winners are much larger than losers — a classic
  trend-following profile. This is *not* a robustness guarantee: a handful of
  big trends carried the result, so the strategy is fragile in a trendless market.
- **The Composite strategy is identical to MA here.** The RSI < 70 entry filter
  was non-binding: on this ticker/period no golden cross coincided with an
  overbought reading, so the added condition provided zero value. Worth testing
  on other assets before calling it a real improvement.
- **MACD underperforms.** Lower return, lower Sharpe, and a deeper -36.8%
  drawdown than MA despite a higher win rate — its wins are too small to offset
  the whipsaws.
- `evaluate.analyze_failures` surfaces each of these automatically instead of
  hiding underperforming periods.

## Roadmap (in development)

- Hybrid retrieval (dense + BM25) and cross-encoder reranking for RAG
- AI agent workflow (planning + tool calling + retry) for end-to-end analysis
- Cost & latency observability for every model/tool call
- Citation verification for LLM-generated reports
- Walk-forward / out-of-sample evaluation
- Parameter optimization and robustness checks
