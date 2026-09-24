"""Execute a plan's steps with dependency ordering and per-step retry."""

from __future__ import annotations

import time

from .retry import TransientError, retry_with_backoff
from .state import AgentState, Plan, Step, StepStatus, ToolResult
from .tool import ToolRegistry, ValidationError


def _resolve(value, results: dict[str, ToolResult]):
    """Resolve ``$step.<id>.data[.<key>...]`` references to concrete values."""
    if isinstance(value, str) and value.startswith("$step."):
        parts = value[len("$step.") :].split(".")
        step_id = parts[0]
        if step_id not in results or not results[step_id].ok:
            raise ValueError(f"Unresolved or failed reference: {value}")
        data = results[step_id].data
        for key in parts[2:]:  # parts[1] is the literal "data" marker.
            if isinstance(data, dict):
                data = data[key]
            else:
                raise ValueError(f"Cannot index into non-dict data for {value}")
        return data
    if isinstance(value, list):
        return [_resolve(item, results) for item in value]
    if isinstance(value, dict):
        return {k: _resolve(v, results) for k, v in value.items()}
    return value


def resolve_args(step: Step, results: dict[str, ToolResult]) -> dict:
    """Substitute ``$step.<id>.data`` references in ``step.args``."""
    return {key: _resolve(value, results) for key, value in step.args.items()}


def topological_waves(steps: list[Step]) -> list[list[Step]]:
    """Group steps into waves such that a step appears after its dependencies."""
    by_id = {step.id: step for step in steps}
    placed: set[str] = set()
    remaining = list(steps)
    waves: list[list[Step]] = []
    while remaining:
        ready = [s for s in remaining if all(dep in placed for dep in s.depends_on)]
        if not ready:
            # Cycle or missing dependency; place the rest sequentially.
            waves.append(remaining)
            break
        waves.append(ready)
        for step in ready:
            placed.add(step.id)
        remaining = [s for s in remaining if s not in ready]
    return waves


def execute_step(
    step: Step,
    registry: ToolRegistry,
    results: dict[str, ToolResult],
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> ToolResult:
    """Execute one step with transient retry; returns a ``ToolResult`` (never raises).

    Args:
        step: The step to execute.
        registry: The tool registry.
        results: Results of already-executed steps (for arg resolution).
        max_attempts: Retry attempts for transient failures.
        base_delay: Initial backoff delay in seconds.
    """
    tool = registry.get(step.tool)
    step.status = StepStatus.RUNNING
    start = time.perf_counter()
    attempts = {"n": 0}
    try:
        args = resolve_args(step, results)

        def call():
            return tool.fn(**args)

        def on_retry(_exc, n):
            attempts["n"] = n

        data = retry_with_backoff(call, max_attempts=max_attempts, base_delay=base_delay, on_retry=on_retry)
        if tool.validator is not None:
            tool.validator(data)
        result = ToolResult(tool=step.tool, ok=True, data=data)
        step.status = StepStatus.OK
    except TransientError as exc:
        result = ToolResult(tool=step.tool, ok=False, error=f"transient: {exc}")
        step.status = StepStatus.FAILED_TRANSIENT
    except ValidationError as exc:
        result = ToolResult(tool=step.tool, ok=False, error=f"validation: {exc}")
        step.status = StepStatus.FAILED_VALIDATION
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors honestly
        result = ToolResult(tool=step.tool, ok=False, error=f"error: {exc}")
        step.status = StepStatus.FAILED_VALIDATION
    result.latency_ms = (time.perf_counter() - start) * 1000.0
    result.retries = attempts["n"]
    step.result = result
    return result


def execute_plan(plan: Plan, registry: ToolRegistry, state: AgentState) -> None:
    """Run all waves sequentially (deterministic and reproducible)."""
    for wave in topological_waves(plan.steps):
        for step in wave:
            state.results[step.id] = execute_step(step, registry, state.results)
