# Tasks

Tracked work items for `stock-trend-lab`. Each item maps to a Git commit.

Legend:

- [ ] Not started
- [~] In progress
- [x] Done

## Phase 0 — Foundation
- [ ] T0.1 Project skeleton (`src/`, `tests/`, `outputs/`, `knowledge/`)
- [ ] T0.2 `requirements.txt`, `.env.example`, logging setup

## Phase 1 — Data
- [ ] T1.1 `data_loader.py` (yfinance + parquet cache + cleaning)

## Phase 2 — Indicators
- [ ] T2.1 `indicators.py` (MA / MACD / RSI / Bollinger / trend state)

## Phase 3 — Strategies & backtest
- [ ] T3.1 `strategy.py` (MA / MACD / Composite)
- [ ] T3.2 `backtest.py` (`run_backtest` + batch)

## Phase 4 — Evaluation
- [ ] T4.1 `evaluate.py` (`compare_strategies` + `analyze_failures`)

## Phase 5 — RAG retrieval stack (pending design)
- [ ] T5.1 embeddings + vector store + chunking
- [ ] T5.2 keyword index + hybrid retrieval (RRF)
- [ ] T5.3 reranker
- [ ] T5.4 indexers (knowledge / news / market-regime)
- [ ] T5.5 citation verification

## Phase 6 — Observability (pending design)
- [ ] T6.1 `UsageTracker` (cost + latency)

## Phase 7 — Agent orchestration (pending design)
- [ ] T7.1 tool registry
- [ ] T7.2 planner + executor + retry
- [ ] T7.3 orchestrator

## Phase 8 — Report
- [ ] T8.1 `report.py` (Markdown + optional LLM/RAG)

## Phase 9 — Tests
- [ ] T9.1 indicator / loader / evaluate / report tests

## Phase 10 — Docs & packaging
- [ ] T10.1 `README.md`
- [ ] T10.2 `docker-compose.yml`, notebook

## Phase 11 — Verification
- [ ] T11.1 install deps, run pytest, real backtest
