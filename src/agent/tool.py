"""Tool abstraction and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ValidationError(Exception):
    """Raised when a tool result fails schema/shape validation (fatal, not transient)."""


@dataclass(frozen=True)
class Tool:
    """A callable exposed to the LLM.

    Attributes:
        name: Unique tool name exposed to the LLM.
        description: Natural-language description the LLM uses to decide when to call it.
        parameters: JSON Schema object describing the tool's arguments.
        fn: The Python callable. Args are passed by keyword.
        validator: Optional result validator; raises ``ValidationError`` on a bad result.
        side_effect_free: If True, the tool may run in parallel with other tools in the same wave.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    validator: Callable[[Any], None] | None = None
    side_effect_free: bool = True


class ToolRegistry:
    """Holds registered tools and renders them in OpenAI's ``tools`` format."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool, rejecting duplicate names."""
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a tool by name, raising KeyError if unknown."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name!r}. Available: {sorted(self._tools)}") from None

    def names(self) -> list[str]:
        """Return the sorted list of registered tool names."""
        return sorted(self._tools)

    def to_openai(self) -> list[dict[str, Any]]:
        """Render all tools in OpenAI chat-completions ``tools`` array format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]
