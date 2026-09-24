"""Agent state and planning data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    FAILED_TRANSIENT = "failed_transient"
    FAILED_VALIDATION = "failed_validation"


@dataclass
class ToolResult:
    """Result of executing a single tool."""

    tool: str
    ok: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    retries: int = 0


@dataclass
class Step:
    """A single planned tool invocation."""

    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    result: ToolResult | None = None


@dataclass
class Plan:
    """The LLM-produced execution plan."""

    goal: str
    steps: list[Step]
    raw: str = ""


@dataclass
class AgentState:
    """Aggregate state for one agent run; feeds observability and reporting."""

    goal: str
    plan: Plan | None = None
    results: dict[str, ToolResult] = field(default_factory=dict)
    report: str = ""
