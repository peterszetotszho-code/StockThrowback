"""Cost and latency tracking for every model/tool call.

Costs are computed from a config-driven pricing table (``config/pricing.json``)
so prices can change without a code release. Local models have no pricing
entry and are recorded as free ($0), while still measuring latency.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DEFAULT_PRICING_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "pricing.json"
)


@dataclass(frozen=True)
class UsageRecord:
    """A single recorded model/tool call.

    Attributes:
        operation: Logical operation name (e.g. "llm.chat", "embed", "rerank").
        model: Model identifier (e.g. "gpt-4.1-mini" or "local:onnx-minilm").
        input_tokens: Input/prompt tokens (0 if unknown or a cache hit).
        output_tokens: Output/completion tokens (0 for embeddings/rerank).
        latency_ms: Wall-clock elapsed milliseconds.
        cost_usd: Cost in USD computed from the pricing table.
        timestamp: Epoch seconds at record time.
        cached: True if served from a local cache (recorded with zero cost).
        extra: Free-form JSON-serializable metadata.
    """

    operation: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)
    cached: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class UsageSpan:
    """Open handle handed out by ``UsageTracker.trace``."""

    def __init__(
        self, operation: str, model: str, cached: bool
    ) -> None:
        self._operation = operation
        self._model = model
        self._cached = cached
        self._input_tokens = 0
        self._output_tokens = 0
        self._extra: dict[str, Any] = {}

    def set_usage(self, input_tokens: int = 0, output_tokens: int = 0, **extra: Any) -> None:
        """Set token counts and optional metadata before the block exits."""
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._extra.update(extra)

    def set_usage_from_openai(self, usage: Any) -> None:
        """Populate tokens from an OpenAI usage object.

        Handles both the Chat Completions shape (``prompt_tokens`` /
        ``completion_tokens``) and the Responses shape (``input_tokens`` /
        ``output_tokens``) so the difference lives in one place.
        """
        if usage is None:
            return
        self._input_tokens = int(
            getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0
        )
        self._output_tokens = int(
            getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0
        )


class UsageTracker:
    """Thread-safe accumulator of cost/latency observations."""

    def __init__(self, pricing_path: str | Path | None = None) -> None:
        self._records: list[UsageRecord] = []
        self._lock = threading.RLock()
        self._pricing = self._load_pricing(pricing_path or DEFAULT_PRICING_PATH)
        self._warned: set[str] = set()

    def _load_pricing(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            logger.warning("Pricing config not found at %s; all costs will be $0.", path)
            return {"per_million_tokens": {}}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _price(self, model: str) -> tuple[float, float]:
        """Return (input_per_1M, output_per_1M) for a model, or (0.0, 0.0)."""
        table = self._pricing.get("per_million_tokens", {})
        entry = table.get(model)
        if entry is None:
            if model not in self._warned:
                logger.warning("No pricing entry for model %r; recording $0 cost.", model)
                self._warned.add(model)
            return (0.0, 0.0)
        return (float(entry.get("input", 0.0)), float(entry.get("output", 0.0)))

    @staticmethod
    def _compute_cost(
        price: tuple[float, float], input_tokens: int, output_tokens: int, cached: bool
    ) -> float:
        if cached:
            return 0.0
        price_in, price_out = price
        return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out

    def record(
        self,
        operation: str,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        cached: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Append one observation; cost is computed here from the pricing table.

        Args:
            operation: Logical operation name.
            model: Model identifier string.
            input_tokens: Input/prompt token count.
            output_tokens: Output/completion token count.
            latency_ms: Elapsed milliseconds (already measured by the caller, or
                by :meth:`trace`).
            cached: Whether the result came from a local cache (cost forced to 0).
            extra: Optional JSON-serializable metadata.

        Returns:
            The recorded ``UsageRecord``.
        """
        cost = self._compute_cost(self._price(model), input_tokens, output_tokens, cached)
        rec = UsageRecord(
            operation,
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            cached=cached,
            extra=dict(extra or {}),
        )
        with self._lock:
            self._records.append(rec)
        return rec

    @contextmanager
    def trace(
        self, operation: str, model: str, *, cached: bool = False
    ) -> Iterator[UsageSpan]:
        """Time a block with ``perf_counter`` and record it on exit.

        Args:
            operation: Logical operation name.
            model: Model identifier string.
            cached: Whether the result is served from a local cache.

        Yields:
            A ``UsageSpan`` whose token usage can be set before the block exits.
        """
        start = time.perf_counter()
        span = UsageSpan(operation, model, cached)
        try:
            yield span
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.record(
                operation,
                model,
                input_tokens=span._input_tokens,
                output_tokens=span._output_tokens,
                latency_ms=latency_ms,
                cached=cached,
                extra=span._extra,
            )

    def reset(self) -> None:
        """Drop all records (e.g. between backtests)."""
        with self._lock:
            self._records.clear()

    def _snapshot(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def to_dict(self) -> dict[str, Any]:
        """Return an aggregated snapshot for serialization."""
        records = self._snapshot()
        by_op: dict[tuple[str, str], dict[str, Any]] = {}
        for r in records:
            key = (r.operation, r.model)
            agg = by_op.setdefault(
                key,
                {
                    "operation": r.operation,
                    "model": r.model,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms_total": 0.0,
                    "cost_usd": 0.0,
                },
            )
            agg["calls"] += 1
            agg["input_tokens"] += r.input_tokens
            agg["output_tokens"] += r.output_tokens
            agg["latency_ms_total"] += r.latency_ms
            agg["cost_usd"] += r.cost_usd

        by_operation = []
        for agg in by_op.values():
            agg["latency_ms_avg"] = agg["latency_ms_total"] / agg["calls"] if agg["calls"] else 0.0
            by_operation.append(agg)
        by_operation.sort(key=lambda a: a["cost_usd"], reverse=True)

        totals = {
            "calls": len(records),
            "input_tokens": sum(r.input_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "latency_ms": sum(r.latency_ms for r in records),
            "cost_usd": sum(r.cost_usd for r in records),
        }
        return {
            "schema_version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "currency": self._pricing.get("currency", "USD"),
            "pricing_schema_version": self._pricing.get("schema_version", 1),
            "totals": totals,
            "by_operation": by_operation,
            "records": [
                {
                    "operation": r.operation,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": round(r.latency_ms, 1),
                    "cost_usd": round(r.cost_usd, 6),
                    "timestamp": r.timestamp,
                    "cached": r.cached,
                    "extra": r.extra,
                }
                for r in records
            ],
        }

    def summary_markdown(self) -> str:
        """Return a Markdown summary table grouped by (operation, model)."""
        data = self.to_dict()
        header = [
            "Operation", "Model", "Calls", "Tokens In", "Tokens Out",
            "Avg Latency (ms)", "Total Latency (ms)", "Cost (USD)",
        ]
        rows = []
        for agg in data["by_operation"]:
            rows.append(
                [
                    agg["operation"],
                    agg["model"],
                    str(agg["calls"]),
                    str(agg["input_tokens"]),
                    str(agg["output_tokens"]),
                    f"{agg['latency_ms_avg']:.1f}",
                    f"{agg['latency_ms_total']:.1f}",
                    f"${agg['cost_usd']:.6f}",
                ]
            )
        totals = data["totals"]
        rows.append(
            [
                "TOTAL", "", str(totals["calls"]), str(totals["input_tokens"]),
                str(totals["output_tokens"]), "-", f"{totals['latency_ms']:.1f}",
                f"${totals['cost_usd']:.6f}",
            ]
        )
        widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]

        def fmt(row: list[str]) -> str:
            return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(header))) + " |"

        sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        lines = ["# Usage Summary (generated " + data["generated_at"] + ")", "", fmt(header), sep]
        lines += [fmt(row) for row in rows]
        return "\n".join(lines)

    def summary_json(self, *, indent: int = 2) -> str:
        """Return the aggregated snapshot as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path) -> None:
        """Write :meth:`summary_json` to a file."""
        Path(path).write_text(self.summary_json(), encoding="utf-8")


_tracker: UsageTracker | None = None
_tracker_lock = threading.Lock()


def get_usage_tracker(pricing_path: str | Path | None = None) -> UsageTracker:
    """Return the process-wide UsageTracker singleton (created lazily)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = UsageTracker(pricing_path=pricing_path)
    return _tracker


def reset_usage_tracker() -> None:
    """Drop the singleton so a fresh instance is created next call (for tests)."""
    global _tracker
    with _tracker_lock:
        _tracker = None
