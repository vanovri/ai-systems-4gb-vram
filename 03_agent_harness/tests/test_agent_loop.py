"""Unit tests for formal bounded loop, guards, trace logging, and OpenAI provider adapter."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from harness.agent import AgentConfig, AgentHarness
from harness.llm import OpenAICompatibleProvider, ScriptedTestProvider
from harness.models import LLMResponse, Message, StopReason, TokenUsage, ToolCall
from harness.tools import ToolRegistry
from harness.trace import TraceLogger


def test_formal_loop_state_transition_and_jsonl_trace(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("MSG = 'old'\n", encoding="utf-8")
    tools = ToolRegistry(tmp_path)
    trace_file = tmp_path / "trace.jsonl"

    seen_memory_lengths: list[int] = []

    scripted_responses = [
        LLMResponse(
            content="List workspace files",
            tool_call=ToolCall(id="c1", name="list_files", arguments={"path": "."}),
            usage=TokenUsage(prompt_tokens=30, completion_tokens=10),
        ),
        LLMResponse(
            content="Update MSG value",
            tool_call=ToolCall(
                id="c2",
                name="edit_file",
                arguments={"path": "hello.py", "old_text": "old", "new_text": "new"},
            ),
            usage=TokenUsage(prompt_tokens=50, completion_tokens=15),
        ),
        LLMResponse(
            content="Done updating hello.py.",
            tool_call=None,
            usage=TokenUsage(prompt_tokens=70, completion_tokens=12),
        ),
    ]
    idx = {"i": 0}

    def callback(messages: list[Message], _schemas: list[dict]) -> LLMResponse:
        seen_memory_lengths.append(len(messages))
        resp = scripted_responses[idx["i"]]
        idx["i"] += 1
        return resp

    harness = AgentHarness(provider=ScriptedTestProvider(callback), tools=tools)
    result = harness.run("Update MSG", trace_path=trace_file, run_id="test-run-1")

    assert result.stop_reason == StopReason.COMPLETED
    assert result.steps_taken == 3
    # m_0 has 2 msgs (system, user); each tool step adds (a_t, o_t) = +2 messages
    assert seen_memory_lengths == [2, 4, 6]
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "MSG = 'new'\n"

    records = TraceLogger.load_jsonl(trace_file)
    assert len(records) == 4  # 3 step events + 1 run_complete
    assert records[0]["tool_call"]["name"] == "list_files"
    assert records[1]["diff"] is not None
    assert "+MSG = 'new'" in records[1]["diff"]
    assert records[-1]["event"] == "run_complete"
    assert records[-1]["stop_reason"] == "completed"


def test_stops_on_max_steps_reached(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 2\n", encoding="utf-8")
    tools = ToolRegistry(tmp_path)

    provider = ScriptedTestProvider(
        [
            LLMResponse(
                content="read a",
                tool_call=ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}),
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            ),
            LLMResponse(
                content="read b",
                tool_call=ToolCall(id="c2", name="read_file", arguments={"path": "b.py"}),
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            ),
            LLMResponse(
                content="never reached",
                tool_call=None,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            ),
        ]
    )

    harness = AgentHarness(provider=provider, tools=tools, config=AgentConfig(max_steps=2))
    res = harness.run("Read forever")
    assert res.stop_reason == StopReason.MAX_STEPS_REACHED
    assert res.steps_taken == 2


def test_doom_loop_detection_halts_repeated_identical_calls(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    tools = ToolRegistry(tmp_path)

    repeated = LLMResponse(
        content="repeat list",
        tool_call=ToolCall(id="c1", name="list_files", arguments={"path": "."}),
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
    )
    provider = ScriptedTestProvider([repeated, repeated, repeated, repeated])

    harness = AgentHarness(
        provider=provider,
        tools=tools,
        config=AgentConfig(max_steps=10, doom_loop_threshold=3),
    )
    res = harness.run("Loop test")
    assert res.stop_reason == StopReason.DOOM_LOOP_DETECTED
    assert res.steps_taken == 3


def test_token_budget_guard_halts_loop(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    tools = ToolRegistry(tmp_path)

    provider = ScriptedTestProvider(
        [
            LLMResponse(
                content="step 1",
                tool_call=ToolCall(id="c1", name="list_files", arguments={"path": "."}),
                usage=TokenUsage(prompt_tokens=400, completion_tokens=200),
            ),
            LLMResponse(
                content="step 2 exceeds budget",
                tool_call=ToolCall(id="c2", name="read_file", arguments={"path": "a.py"}),
                usage=TokenUsage(prompt_tokens=500, completion_tokens=100),
            ),
        ]
    )
    harness = AgentHarness(
        provider=provider,
        tools=tools,
        config=AgentConfig(max_steps=10, max_token_budget=1000),
    )
    res = harness.run("Budget test")
    assert res.stop_reason == StopReason.TOKEN_BUDGET_EXCEEDED
    assert res.total_usage.total_tokens == 1200


def test_unknown_tool_stops_cleanly(tmp_path: Path) -> None:
    tools = ToolRegistry(tmp_path)
    provider = ScriptedTestProvider(
        [
            LLMResponse(
                content="Try shell",
                tool_call=ToolCall(id="c1", name="bash_exec", arguments={"cmd": "whoami"}),
                usage=TokenUsage(prompt_tokens=15, completion_tokens=10),
            )
        ]
    )
    harness = AgentHarness(provider=provider, tools=tools)
    res = harness.run("Unauthorized tool test")
    assert res.stop_reason == StopReason.UNKNOWN_TOOL
    assert res.steps[0].tool_result is not None
    assert res.steps[0].tool_result.error_code == "UNKNOWN_TOOL"


def test_openai_compatible_provider_parses_tool_calls_with_mock_transport() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Let me inspect files.",
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": '{"path": "."}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 111, "completion_tokens": 22},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:20128/v1",
        api_key="test-token-not-real",
        model="local-coder",
        client=client,
    )
    resp = provider.complete([Message(role="user", content="Fix bug")], tools=[])

    assert captured["url"] == "http://127.0.0.1:20128/v1/chat/completions"
    assert captured["auth"] == "Bearer test-token-not-real"
    assert captured["body"]["model"] == "local-coder"
    assert resp.tool_call is not None
    assert resp.tool_call.name == "list_files"
    assert resp.tool_call.arguments == {"path": "."}
    assert resp.usage.total_tokens == 133
