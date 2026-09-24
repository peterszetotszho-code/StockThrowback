"""LLM abstraction so OpenAI is swappable and tests can inject a fake."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .retry import TransientError


@dataclass
class ToolCall:
    """A parsed tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized LLM completion result."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Any = None


class ChatModel(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...


class FakeChatModel:
    """Deterministic stub: replays a scripted list of responses, then repeats the last."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self._i = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        response = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return response


class OpenAIChatModel:
    """Thin wrapper over the OpenAI SDK; works with any OpenAI-compatible API.

    Pass ``base_url`` and ``api_key`` to target a compatible provider such as
    DeepSeek; otherwise the OpenAI defaults (env vars) are used.
    """

    def __init__(
        self,
        model: str,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self._base_url = base_url
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # Lazy import.

            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
    ) -> LLMResponse:
        from openai import APIConnectionError, APIStatusError  # Lazy import.

        from ..observability.usage_tracker import get_usage_tracker

        tracker = get_usage_tracker()
        try:
            client = self._get_client()
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            with tracker.trace("llm.chat", self.model) as span:
                resp = client.chat.completions.create(**kwargs)
                span.set_usage_from_openai(resp.usage)
        except (APIConnectionError, APIStatusError) as exc:
            # Network / 429 / 5xx are transient; retryable upstream.
            raise TransientError(f"LLM API error: {exc}") from exc

        message = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return LLMResponse(content=message.content, tool_calls=tool_calls, usage=resp.usage)
