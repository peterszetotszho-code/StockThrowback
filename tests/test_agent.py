"""Tests for the agent orchestration layer."""

import pytest

from src.agent.executor import execute_step, resolve_args, topological_waves
from src.agent.llm import FakeChatModel, LLMResponse
from src.agent.orchestrator import run_agent
from src.agent.retry import TransientError, retry_with_backoff
from src.agent.state import Step, ToolResult
from src.agent.tool import Tool, ToolRegistry, ValidationError


def test_retry_with_backoff_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("boom")
        return "ok"

    assert retry_with_backoff(flaky, max_attempts=3, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_with_backoff_exhausts():
    def always_fail():
        raise TransientError("boom")

    with pytest.raises(TransientError):
        retry_with_backoff(always_fail, max_attempts=2, base_delay=0.0)


def test_topological_waves_diamond():
    steps = [
        Step(id="s1", tool="a", args={}),
        Step(id="s2", tool="b", args={}),
        Step(id="s3", tool="c", args={}, depends_on=["s1", "s2"]),
    ]
    waves = topological_waves(steps)
    assert {s.id for s in waves[0]} == {"s1", "s2"}
    assert [s.id for s in waves[1]] == ["s3"]


def test_resolve_args_substitution():
    results = {"s1": ToolResult(tool="fetch", ok=True, data={"price": 10})}
    step = Step(id="s2", tool="t", args={"x": "$step.s1.data.price"})
    assert resolve_args(step, results) == {"x": 10}


def test_registry_duplicate_and_unknown():
    registry = ToolRegistry()
    registry.register(Tool("a", "desc", {}, fn=lambda: 1))
    with pytest.raises(ValueError):
        registry.register(Tool("a", "desc", {}, fn=lambda: 2))
    with pytest.raises(KeyError):
        registry.get("missing")


def test_execute_step_retries_transient():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TransientError("network")
        return 42

    registry = ToolRegistry()
    registry.register(Tool("flaky", "desc", {}, fn=flaky))
    step = Step(id="s", tool="flaky", args={})
    result = execute_step(step, registry, {}, max_attempts=3, base_delay=0.0)
    assert result.ok
    assert result.data == 42
    assert result.retries == 1


def test_execute_step_validation_error_is_fatal():
    def reject(_data):
        raise ValidationError("empty result")

    registry = ToolRegistry()
    registry.register(Tool("bad", "desc", {}, fn=lambda: [], validator=reject))
    step = Step(id="s", tool="bad", args={})
    result = execute_step(step, registry, {}, max_attempts=3, base_delay=0.0)
    assert not result.ok
    assert "validation" in result.error


def test_run_agent_end_to_end():
    registry = ToolRegistry()
    registry.register(
        Tool(
            "double",
            "double x",
            {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
            fn=lambda x: x * 2,
        )
    )
    registry.register(
        Tool(
            "add",
            "add a and b",
            {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
            fn=lambda a, b: a + b,
        )
    )

    llm = FakeChatModel(
        [
            LLMResponse(
                content='[{"id":"s1","tool":"double","args":{"x":3}},'
                '{"id":"s2","tool":"add","args":{"a":"$step.s1.data","b":1}}]'
            ),
            LLMResponse(content="All done."),
        ]
    )
    state = run_agent("compute 2*3 + 1", llm, registry)
    assert state.results["s1"].data == 6
    assert state.results["s2"].data == 7
    assert state.report == "All done."
    assert len(llm.calls) == 2
