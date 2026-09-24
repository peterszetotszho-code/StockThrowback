# Tasks

Tracked work items for `stock-trend-lab`. Each item maps to a Git commit.

Legend:

- [ ] Not started
- [~] In progress
- [x] Done

## Phase 0 — Foundation
- [x] T0.1 Project skeleton (`src/`, `tests/`, `outputs/`, `knowledge/`)
- [x] T0.2 `requirements.txt`, `.env.example`, logging setup

## Phase 1 — Data
- [x] T1.1 `data_loader.py` (yfinance + parquet cache + cleaning)

## Phase 2 — Indicators
- [x] T2.1 `indicators.py` (MA / MACD / RSI / Bollinger / trend state)

## Phase 3 — Strategies & backtest
- [x] T3.1 `strategy.py` (MA / MACD / Composite)
- [x] T3.2 `backtest.py` (`run_backtest` + batch)

## Phase 4 — Evaluation
- [x] T4.1 `evaluate.py` (`compare_strategies` + `analyze_failures`)

## Phase 5 — RAG retrieval stack (design ready)
- [x] T5.1 embeddings + vector store + chunking
- [x] T5.2 keyword index + hybrid retrieval (RRF)
- [x] T5.3 reranker
- [x] T5.4 indexers (knowledge / news / market-regime)
- [x] T5.5 citation verification

## Phase 6 — Observability (design ready)
- [x] T6.1 `UsageTracker` (cost + latency)

## Phase 7 — Agent orchestration (design ready)
- [x] T7.1 tool registry
- [x] T7.2 planner + executor + retry
- [x] T7.3 orchestrator

## Phase 8 — Report
- [x] T8.1 `report.py` (Markdown core + RAG pipeline report)

## Phase 12 — Integration (wiring)
- [x] T12.1 wire UsageTracker into agent LLM calls + pipeline
- [x] T12.2 end-to-end RAG report endpoint (`/api/report`)

## Phase 13 — Robustness
- [x] T13.1 walk-forward / out-of-sample evaluation
- [x] T13.2 parameter optimization
- [x] T13.3 Monte Carlo robustness check

## Phase 14 — RAG report depth
- [x] T14.1 news event recall (GDELT) for worst drawdown
- [x] T14.2 knowledge-base retrieval into report

## Phase 9 — Tests
- [x] T9.1 indicator / loader / evaluate / report tests

## Phase 10 — Docs & packaging
- [x] T10.1 `README.md`
- [x] T10.2 `docker-compose.yml`, notebook

## Phase 11 — Verification
- [x] T11.1 install deps, run pytest (47 passed), real backtest + README results
