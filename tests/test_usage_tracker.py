"""Tests for the cost/latency usage tracker."""

import json

import pytest

from src.observability.usage_tracker import UsageTracker


def _write_pricing(tmp_path):
    data = {
        "schema_version": 1,
        "currency": "USD",
        "per_million_tokens": {
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "text-embedding-3-small": {"input": 0.02, "output": 0.00},
        },
    }
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_record_computes_cost(tmp_path):
    tracker = UsageTracker(pricing_path=_write_pricing(tmp_path))
    rec = tracker.record("llm.chat", "gpt-4.1-mini", input_tokens=1000, output_tokens=1000)
    # (1000/1e6)*0.40 + (1000/1e6)*1.60 = 0.0004 + 0.0016 = 0.002
    assert rec.cost_usd == pytest.approx(0.002)


def test_missing_model_is_free(tmp_path):
    tracker = UsageTracker(pricing_path=_write_pricing(tmp_path))
    rec = tracker.record("embed", "local:onnx-minilm", input_tokens=500)
    assert rec.cost_usd == 0.0


def test_cached_is_free(tmp_path):
    tracker = UsageTracker(pricing_path=_write_pricing(tmp_path))
    rec = tracker.record("embed", "text-embedding-3-small", input_tokens=500, cached=True)
    assert rec.cost_usd == 0.0


def test_trace_measures_latency(tmp_path):
    tracker = UsageTracker(pricing_path=_write_pricing(tmp_path))
    with tracker.trace("embed", "text-embedding-3-small") as span:
        span.set_usage(input_tokens=128)
    rec = tracker._snapshot()[0]
    assert rec.input_tokens == 128
    assert rec.latency_ms >= 0


def test_summary_markdown_has_total(tmp_path):
    tracker = UsageTracker(pricing_path=_write_pricing(tmp_path))
    tracker.record("llm.chat", "gpt-4.1-mini", input_tokens=1000, output_tokens=1000)
    md = tracker.summary_markdown()
    assert "TOTAL" in md
    assert "gpt-4.1-mini" in md
    assert "Cost (USD)" in md


def test_summary_json_roundtrip(tmp_path):
    tracker = UsageTracker(pricing_path=_write_pricing(tmp_path))
    tracker.record("llm.chat", "gpt-4.1-mini", input_tokens=10, output_tokens=10)
    data = json.loads(tracker.summary_json())
    assert data["totals"]["calls"] == 1
    assert data["totals"]["input_tokens"] == 10
