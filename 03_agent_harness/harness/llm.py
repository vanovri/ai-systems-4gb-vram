"""Provider adapters: OpenAI-compatible live endpoint, Scripted replay, and Offline heuristic."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from harness.models import LLMResponse, Message, TokenUsage, ToolCall


def estimate_tokens(text: str) -> int:
    """Lightweight char-to-token estimate when provider omits token usage."""
    return max(1, len(text) // 4) if text else 0


class LLMProvider(ABC):
    """Minimal provider contract consumed by AgentHarness."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Return next thought + optional tool_call given memory state m_t."""


class OpenAICompatibleProvider(LLMProvider):
    """Zero-bloat OpenAI-compatible chat completions adapter (OpenAI / 9Router / Gemini)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "assistant" and msg.tool_call is not None:
                serialized.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": msg.tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": msg.tool_call.name,
                                    "arguments": json.dumps(msg.tool_call.arguments),
                                },
                            }
                        ],
                    }
                )
            elif msg.role == "tool":
                serialized.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "call_1",
                        "name": msg.tool_name or "tool",
                        "content": msg.content,
                    }
                )
            else:
                serialized.append({"role": msg.role, "content": msg.content})
        return serialized

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.0,
        }

        url = f"{self.base_url}/chat/completions"
        if self._client is not None:
            resp = self._client.post(url, headers=headers, json=payload)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_call: ToolCall | None = None
        raw_calls = choice.get("tool_calls") or []
        if raw_calls:
            first = raw_calls[0]
            fn = first["function"]
            raw_args = fn.get("arguments", "{}")
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            tool_call = ToolCall(
                id=first.get("id", "call_1"),
                name=fn["name"],
                arguments=parsed_args or {},
            )

        raw_usage = data.get("usage") or {}
        if raw_usage:
            usage = TokenUsage(
                prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
                completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            )
        else:
            prompt_est = sum(estimate_tokens(m.content) for m in messages)
            comp_est = estimate_tokens(content) + (
                estimate_tokens(json.dumps(tool_call.arguments)) if tool_call else 0
            )
            usage = TokenUsage(prompt_tokens=prompt_est, completion_tokens=comp_est)

        return LLMResponse(
            content=content,
            tool_call=tool_call,
            usage=usage,
            model=data.get("model", self.model),
        )


class ScriptedTestProvider(LLMProvider):
    """Deterministic replay provider for CI and unit tests."""

    def __init__(
        self,
        responses: Sequence[LLMResponse] | Callable[[list[Message], list[dict[str, Any]]], LLMResponse],
    ) -> None:
        self._responses = list(responses) if not callable(responses) else None
        self._callback = responses if callable(responses) else None
        self._index = 0

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        if self._callback is not None:
            return self._callback(messages, tools)
        assert self._responses is not None
        if self._index >= len(self._responses):
            return LLMResponse(
                content="No further scripted responses; finishing.",
                tool_call=None,
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10),
                model="scripted-test",
            )
        response = self._responses[self._index]
        self._index += 1
        return response


class HeuristicOfflineProvider(LLMProvider):
    """Observation-driven offline coding agent for deterministic demos & benchmark eval.

    Inspects the workspace via list_files -> run_tests -> read_file -> edit_file -> run_tests,
    reacting strictly to tool observations in memory state m_t.
    """

    def __init__(self) -> None:
        self._call_counter = 0

    def _next_id(self) -> str:
        self._call_counter += 1
        return f"offline_call_{self._call_counter}"

    def _usage(self, messages: list[Message], out_text: str) -> TokenUsage:
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        return TokenUsage(
            prompt_tokens=max(16, prompt_tokens),
            completion_tokens=max(12, estimate_tokens(out_text)),
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        tool_msgs = [m for m in messages if m.role == "tool"]

        # 1. Discover files first
        listed_files: list[str] = []
        for msg in tool_msgs:
            if msg.tool_name == "list_files":
                body = re.sub(r"^\[ok=(?:True|False)\]\s*", "", msg.content)
                listed_files = [
                    line.strip()
                    for line in body.splitlines()
                    if line.strip() and not line.startswith("(")
                ]

        if not listed_files:
            thought = "Inspect workspace structure using list_files."
            return LLMResponse(
                content=thought,
                tool_call=ToolCall(id=self._next_id(), name="list_files", arguments={"path": "."}),
                usage=self._usage(messages, thought),
                model="heuristic-offline-v1",
            )

        # 2. Check test execution observations
        test_msgs = [m for m in tool_msgs if m.tool_name == "run_tests"]
        if not test_msgs:
            thought = "Run test suite to reproduce current failures."
            return LLMResponse(
                content=thought,
                tool_call=ToolCall(
                    id=self._next_id(), name="run_tests", arguments={"command": "pytest -q"}
                ),
                usage=self._usage(messages, thought),
                model="heuristic-offline-v1",
            )

        last_test = test_msgs[-1]
        if "[ok=True]" in last_test.content:
            thought = (
                "All unit tests pass after applying the targeted fix. "
                "Verification complete; stopping agent loop."
            )
            return LLMResponse(
                content=thought,
                tool_call=None,
                usage=self._usage(messages, thought),
                model="heuristic-offline-v1",
            )

        # 3. Read source files that haven't been read yet
        read_paths: dict[str, str] = {}
        for msg in tool_msgs:
            if msg.tool_name == "read_file":
                match = re.search(r"# ([^\s]+)\s+\(lines", msg.content)
                if match:
                    read_paths[match.group(1)] = msg.content

        src_files = [
            f for f in listed_files if f.endswith(".py") and not f.startswith("tests/") and "/test_" not in f
        ]
        for src in src_files:
            if src not in read_paths:
                thought = f"Read implementation file '{src}' to diagnose failing tests."
                return LLMResponse(
                    content=thought,
                    tool_call=ToolCall(
                        id=self._next_id(),
                        name="read_file",
                        arguments={"path": src, "start_line": 1, "end_line": 200},
                    ),
                    usage=self._usage(messages, thought),
                    model="heuristic-offline-v1",
                )

        # 4. Diagnose known bug signatures in read source files and apply targeted patch
        edited_already = any(m.tool_name == "edit_file" and "[ok=True]" in m.content for m in tool_msgs)
        if not edited_already:
            for src, content in read_paths.items():
                patch = self._propose_patch(src, content)
                if patch is not None:
                    old_text, new_text, rationale = patch
                    return LLMResponse(
                        content=rationale,
                        tool_call=ToolCall(
                            id=self._next_id(),
                            name="edit_file",
                            arguments={"path": src, "old_text": old_text, "new_text": new_text},
                        ),
                        usage=self._usage(messages, rationale + new_text),
                        model="heuristic-offline-v1",
                    )

        # 5. Re-run tests after edit
        if edited_already and tool_msgs[-1].tool_name == "edit_file":
            thought = "Re-run pytest to verify the fix resolves all failing tests."
            return LLMResponse(
                content=thought,
                tool_call=ToolCall(
                    id=self._next_id(), name="run_tests", arguments={"command": "pytest -q"}
                ),
                usage=self._usage(messages, thought),
                model="heuristic-offline-v1",
            )

        # Fallback: stop gracefully if unable to diagnose further
        thought = "Unable to derive another safe automated fix; stopping cleanly."
        return LLMResponse(
            content=thought,
            tool_call=None,
            usage=self._usage(messages, thought),
            model="heuristic-offline-v1",
        )

    @staticmethod
    def _propose_patch(path: str, numbered_content: str) -> tuple[str, str, str] | None:
        # Bug 1: checkout_cart discount percentage subtracted literally instead of scaled
        if "return total - percent" in numbered_content:
            return (
                "    return total - percent",
                "    return round(total * (1 - percent / 100), 2)",
                f"Found percentage discount bug in {path}: subtracting raw percent instead of multiplying by (1 - percent / 100).",
            )

        # Bug 2: sliding window rate limiter off-by-one capacity check (`>` instead of `>=`)
        if "if len(bucket) > self.max_requests:" in numbered_content:
            return (
                "        if len(bucket) > self.max_requests:",
                "        if len(bucket) >= self.max_requests:",
                f"Found off-by-one bug in {path}: bucket capacity uses '>' instead of '>=' allowing max_requests + 1.",
            )

        # Bug 3: CSV parser naive comma split breaking quoted commas and escaped double quotes
        if "return [cell.strip('\"') for cell in line.split(\",\")]" in numbered_content:
            old_block = (
                "def parse_csv_line(line: str) -> list[str]:\n"
                "    \"\"\"Parse a single RFC-4180 CSV line into fields.\"\"\"\n"
                "    if line == \"\":\n"
                "        return []\n"
                "    return [cell.strip('\"') for cell in line.split(\",\")]"
            )
            new_block = (
                "def parse_csv_line(line: str) -> list[str]:\n"
                "    \"\"\"Parse a single RFC-4180 CSV line into fields.\"\"\"\n"
                "    if line == \"\":\n"
                "        return []\n"
                "    fields: list[str] = []\n"
                "    buf: list[str] = []\n"
                "    in_quotes = False\n"
                "    i = 0\n"
                "    n = len(line)\n"
                "    while i < n:\n"
                "        ch = line[i]\n"
                "        if in_quotes:\n"
                "            if ch == '\"':\n"
                "                if i + 1 < n and line[i + 1] == '\"':\n"
                "                    buf.append('\"')\n"
                "                    i += 1\n"
                "                else:\n"
                "                    in_quotes = False\n"
                "            else:\n"
                "                buf.append(ch)\n"
                "        else:\n"
                "            if ch == '\"':\n"
                "                in_quotes = True\n"
                "            elif ch == ',':\n"
                "                fields.append(\"\".join(buf))\n"
                "                buf = []\n"
                "            else:\n"
                "                buf.append(ch)\n"
                "        i += 1\n"
                "    fields.append(\"\".join(buf))\n"
                "    return fields"
            )
            return (
                old_block,
                new_block,
                f"Found CSV parsing bug in {path}: naive split(',') fails on quoted commas and escaped double quotes.",
            )

        return None
