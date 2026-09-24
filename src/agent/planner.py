"""Turn a goal into a concrete plan via the LLM."""

from __future__ import annotations

import json
import re

from .prompts import PLAN_PROMPT
from .state import Plan, Step
from .tool import ToolRegistry


def build_plan(goal: str, llm, registry: ToolRegistry, max_replans: int = 2) -> Plan:
    """Ask the LLM for a step-by-step JSON plan, then parse and validate it.

    Args:
        goal: The natural-language goal.
        llm: A ``ChatModel`` instance.
        registry: The tool registry the plan's tool names are checked against.
        max_replans: Maximum retries when the returned plan is invalid.

    Returns:
        A ``Plan`` with validated steps.

    Raises:
        ValueError: If no valid plan is produced after ``max_replans`` retries.
    """
    tool_names = ", ".join(registry.names())
    messages = [{"role": "system", "content": PLAN_PROMPT.format(tools=tool_names, goal=goal)}]
    for _ in range(max_replans + 1):
        response = llm.complete(messages, temperature=0.0)
        content = response.content or ""
        steps = _parse_steps(content, registry)
        if steps is not None:
            return Plan(goal=goal, steps=steps, raw=content)
        messages.append(
            {
                "role": "user",
                "content": "That plan was not valid JSON or referenced an unknown tool. "
                "Return only a JSON array of step objects.",
            }
        )
    raise ValueError(f"Planning failed after {max_replans} replans for goal: {goal!r}")


def _parse_steps(content: str, registry: ToolRegistry) -> list[Step] | None:
    data = _extract_json(content)
    if not isinstance(data, list):
        return None
    steps: list[Step] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        name = item.get("tool")
        try:
            registry.get(name)
        except (KeyError, TypeError):
            return None
        steps.append(
            Step(
                id=str(item.get("id", f"step{len(steps)}")),
                tool=name,
                args=item.get("args") or {},
                depends_on=item.get("depends_on") or [],
                description=item.get("description", ""),
            )
        )
    return steps


def _extract_json(text: str):
    """Best-effort extraction of a JSON value from LLM output."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
