"""Typed state and protocol models for the agent harness."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class StopReason(str, Enum):
    """Formal termination conditions for the bounded agent loop."""

    COMPLETED = "completed"
    UNKNOWN_TOOL = "unknown_tool"
    MAX_STEPS_REACHED = "max_steps_reached"
    DOOM_LOOP_DETECTED = "doom_loop_detected"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"


class TokenUsage(BaseModel):
    """Prompt and completion token accounting for budget enforcement."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class ToolCall(BaseModel):
    """Structured action a_t emitted by the model."""

    id: str = "call_1"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Structured observation o_t returned by a sandboxed tool."""

    tool_name: str
    ok: bool
    output: str
    error_code: str | None = None
    diff: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Element of memory state m_t."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call: ToolCall | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None


class LLMResponse(BaseModel):
    """Normalized model output for one control-loop iteration."""

    content: str = ""
    tool_call: ToolCall | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = "unknown"


class AgentStep(BaseModel):
    """Single logged transition m_{t+1} = m_t (+) (a_t, o_t)."""

    step: int
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    thought: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    duration_ms: float = 0.0
    diff: str | None = None
    step_tokens: TokenUsage = Field(default_factory=TokenUsage)
    cumulative_tokens: int = 0


class AgentRunResult(BaseModel):
    """Final outcome of an AgentHarness execution."""

    run_id: str
    task: str
    workspace: str
    stop_reason: StopReason
    final_answer: str
    steps_taken: int
    total_usage: TokenUsage
    total_duration_ms: float
    steps: list[AgentStep] = Field(default_factory=list)
    trace_path: str | None = None
