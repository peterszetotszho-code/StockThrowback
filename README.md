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

_To be reproduced._ This section is intentionally a placeholder: the project
never ships fabricated numbers. Run the Quick Start commands against real
market data and paste the resulting comparison table here, for example:

| Strategy | Return [%] | Sharpe Ratio | Max. Drawdown [%] | Win Rate [%] | Trades |
|---|---|---|---|---|---|
| _(fill in after a real run)_ | | | | | |

## Failure Analysis

Honest failure analysis is a first-class deliverable, not an afterthought.
`evaluate.analyze_failures` reports losing trades, the deepest drawdown window,
and a qualitative caveat (e.g. whipsaw in ranging markets) instead of hiding
underperforming periods. Record your own findings here after a real run.

## Roadmap (in development)

- Hybrid retrieval (dense + BM25) and cross-encoder reranking for RAG
- AI agent workflow (planning + tool calling + retry) for end-to-end analysis
- Cost & latency observability for every model/tool call
- Citation verification for LLM-generated reports
- Walk-forward / out-of-sample evaluation
- Parameter optimization and robustness checks
