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
- [ ] T5.1 embeddings + vector store + chunking
- [ ] T5.2 keyword index + hybrid retrieval (RRF)
- [ ] T5.3 reranker
- [ ] T5.4 indexers (knowledge / news / market-regime)
- [ ] T5.5 citation verification

## Phase 6 — Observability (design ready)
- [ ] T6.1 `UsageTracker` (cost + latency)

## Phase 7 — Agent orchestration (design ready)
- [ ] T7.1 tool registry
- [ ] T7.2 planner + executor + retry
- [ ] T7.3 orchestrator

## Phase 8 — Report
- [~] T8.1 `report.py` (Markdown core done; LLM/RAG path pending Phase 5-7)

## Phase 9 — Tests
- [x] T9.1 indicator / loader / evaluate / report tests

## Phase 10 — Docs & packaging
- [x] T10.1 `README.md`
- [x] T10.2 `docker-compose.yml`, notebook

## Phase 11 — Verification
- [~] T11.1 install deps + run pytest (15 passed); real backtest + README results pending
