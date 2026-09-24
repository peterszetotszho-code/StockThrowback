"""Top-level agent loop: plan -> execute -> report."""

from __future__ import annotations

from .executor import execute_plan
from .planner import build_plan
from .prompts import REPORT_PROMPT
from .state import AgentState
from .tool import ToolRegistry


def run_agent(
    goal: str,
    llm,
    registry: ToolRegistry,
    *,
    max_replans: int = 2,
) -> AgentState:
    """Run the full workflow and return the final state (report + trace).

    Args:
        goal: The natural-language goal to accomplish.
        llm: A ``ChatModel`` instance (real or fake).
        registry: The tool registry.
        max_replans: Maximum plan retries before giving up.

    Returns:
        An ``AgentState`` holding the plan, per-step results, and the report.
    """
    state = AgentState(goal=goal)
    state.plan = build_plan(goal, llm, registry, max_replans=max_replans)
    execute_plan(state.plan, registry, state)

    summary = _summarize(state)
    report_response = llm.complete(
        [{"role": "user", "content": REPORT_PROMPT.format(goal=goal, summary=summary)}],
        temperature=0.0,
    )
    state.report = report_response.content or ""
    return state


def _summarize(state: AgentState) -> str:
    """Render a compact, honest digest of every step's outcome."""
    lines: list[str] = []
    for step in state.plan.steps:
        result = state.results.get(step.id)
        line = f"- {step.id} ({step.tool}): {step.status.value}"
        if result is not None and not result.ok:
            line += f" -> {result.error}"
        lines.append(line)
    return "\n".join(lines)
