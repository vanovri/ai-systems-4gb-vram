"""Bounded coding-agent control loop implementing m_{t+1} = m_t (+) (a_t, o_t)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from harness.llm import LLMProvider
from harness.models import (
    AgentRunResult,
    AgentStep,
    Message,
    StopReason,
    TokenUsage,
    ToolCall,
)
from harness.tools import ToolRegistry
from harness.trace import TraceLogger

DEFAULT_SYSTEM_PROMPT = (
    "You are a minimal coding agent running inside a sandboxed workspace. "
    "You have exactly 4 tools: list_files, read_file, edit_file, run_tests. "
    "Diagnose bugs systematically, apply minimal exact edits, verify with run_tests, "
    "and respond with a concise final summary once tests pass."
)


@dataclass(slots=True)
class AgentConfig:
    """Bounded control loop configuration and safety budgets."""

    max_steps: int = 12
    doom_loop_threshold: int = 3
    max_token_budget: int = 64_000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    stop_on_unknown_tool: bool = True


class AgentHarness:
    """Formal bounded control loop:

        m_{t+1} = m_t (+) (a_t, o_t)

    Terminates cleanly when:
      1. a_t is not in tool set T (final natural-language answer or unknown tool)
      2. step >= max_steps (T_max)
      3. identical (tool_name, arguments) repeats >= doom_loop_threshold
      4. cumulative tokens exceed max_token_budget
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.config = config or AgentConfig()

    @staticmethod
    def _action_signature(call: ToolCall) -> str:
        canonical_args = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        return f"{call.name}:{canonical_args}"

    def run(
        self,
        task: str,
        *,
        trace_path: Path | str | None = None,
        run_id: str | None = None,
    ) -> AgentRunResult:
        run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        logger = TraceLogger(trace_path=trace_path, run_id=run_id)
        tool_schemas = self.tools.openai_schemas()

        # Memory state m_0
        memory: list[Message] = [
            Message(role="system", content=self.config.system_prompt),
            Message(role="user", content=task),
        ]

        steps: list[AgentStep] = []
        total_usage = TokenUsage()
        signature_counts: dict[str, int] = {}
        stop_reason = StopReason.MAX_STEPS_REACHED
        final_answer = ""
        run_start = time.perf_counter()

        for step_idx in range(1, self.config.max_steps + 1):
            step_start = time.perf_counter()
            response = self.provider.complete(memory, tool_schemas)
            total_usage = total_usage + response.usage

            # Token budget guard
            if total_usage.total_tokens > self.config.max_token_budget:
                step_record = AgentStep(
                    step=step_idx,
                    thought=response.content,
                    tool_call=response.tool_call,
                    tool_result=None,
                    duration_ms=(time.perf_counter() - step_start) * 1000.0,
                    step_tokens=response.usage,
                    cumulative_tokens=total_usage.total_tokens,
                )
                steps.append(step_record)
                logger.log_step(step_record)
                stop_reason = StopReason.TOKEN_BUDGET_EXCEEDED
                final_answer = (
                    f"Stopped: token budget exceeded ({total_usage.total_tokens} > "
                    f"{self.config.max_token_budget})."
                )
                break

            action = response.tool_call

            # Termination condition: a_t not in T (natural-language completion)
            if action is None:
                memory.append(Message(role="assistant", content=response.content))
                step_record = AgentStep(
                    step=step_idx,
                    thought=response.content,
                    tool_call=None,
                    tool_result=None,
                    duration_ms=(time.perf_counter() - step_start) * 1000.0,
                    step_tokens=response.usage,
                    cumulative_tokens=total_usage.total_tokens,
                )
                steps.append(step_record)
                logger.log_step(step_record)
                stop_reason = StopReason.COMPLETED
                final_answer = response.content
                break

            # Termination condition: a_t names an unknown tool not in T
            if not self.tools.has_tool(action.name):
                observation = self.tools.execute(action.name, action.arguments)
                memory.append(Message(role="assistant", content=response.content, tool_call=action))
                memory.append(
                    Message(
                        role="tool",
                        tool_call_id=action.id,
                        tool_name=action.name,
                        content=f"[ok={observation.ok}] {observation.output}",
                    )
                )
                step_record = AgentStep(
                    step=step_idx,
                    thought=response.content,
                    tool_call=action,
                    tool_result=observation,
                    duration_ms=(time.perf_counter() - step_start) * 1000.0,
                    step_tokens=response.usage,
                    cumulative_tokens=total_usage.total_tokens,
                )
                steps.append(step_record)
                logger.log_step(step_record)
                if self.config.stop_on_unknown_tool:
                    stop_reason = StopReason.UNKNOWN_TOOL
                    final_answer = f"Stopped: unknown tool '{action.name}' not in T."
                    break
                continue

            # Doom-loop detection before re-executing identical tool call
            sig = self._action_signature(action)
            signature_counts[sig] = signature_counts.get(sig, 0) + 1
            if signature_counts[sig] >= self.config.doom_loop_threshold:
                step_record = AgentStep(
                    step=step_idx,
                    thought=response.content,
                    tool_call=action,
                    tool_result=None,
                    duration_ms=(time.perf_counter() - step_start) * 1000.0,
                    step_tokens=response.usage,
                    cumulative_tokens=total_usage.total_tokens,
                )
                steps.append(step_record)
                logger.log_step(step_record)
                stop_reason = StopReason.DOOM_LOOP_DETECTED
                final_answer = (
                    f"Stopped: doom-loop detected ({action.name} repeated "
                    f"{signature_counts[sig]} times with identical arguments)."
                )
                break

            # Execute action a_t in sandbox -> observation o_t
            observation = self.tools.execute(action.name, action.arguments)

            # Reset non-mutating repetition counts after a successful workspace edit
            if action.name == "edit_file" and observation.ok:
                signature_counts.clear()

            # State transition: m_{t+1} = m_t (+) (a_t, o_t)
            memory.append(Message(role="assistant", content=response.content, tool_call=action))
            memory.append(
                Message(
                    role="tool",
                    tool_call_id=action.id,
                    tool_name=action.name,
                    content=f"[ok={observation.ok}] {observation.output}",
                )
            )

            step_record = AgentStep(
                step=step_idx,
                thought=response.content,
                tool_call=action,
                tool_result=observation,
                duration_ms=(time.perf_counter() - step_start) * 1000.0,
                diff=observation.diff,
                step_tokens=response.usage,
                cumulative_tokens=total_usage.total_tokens,
            )
            steps.append(step_record)
            logger.log_step(step_record)
        else:
            stop_reason = StopReason.MAX_STEPS_REACHED
            final_answer = f"Stopped: reached max_steps ({self.config.max_steps})."

        total_duration_ms = (time.perf_counter() - run_start) * 1000.0
        logger.log_summary(
            task=task,
            workspace=str(self.tools.sandbox.root),
            stop_reason=stop_reason,
            steps_taken=len(steps),
            total_usage=total_usage,
            total_duration_ms=total_duration_ms,
            final_answer=final_answer,
        )

        return AgentRunResult(
            run_id=run_id,
            task=task,
            workspace=str(self.tools.sandbox.root),
            stop_reason=stop_reason,
            final_answer=final_answer,
            steps_taken=len(steps),
            total_usage=total_usage,
            total_duration_ms=round(total_duration_ms, 3),
            steps=steps,
            trace_path=str(logger.trace_path) if logger.trace_path else None,
        )
