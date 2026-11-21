"""Structured JSONL trace logging for agent harness steps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.models import AgentStep, StopReason, TokenUsage


class TraceLogger:
    """Appends structured JSONL step records (tool_call, tool_result, duration_ms, diff)."""

    def __init__(self, trace_path: Path | str | None, run_id: str) -> None:
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.run_id = run_id
        self.records: list[dict[str, Any]] = []
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            # Reset trace file for this run
            self.trace_path.write_text("", encoding="utf-8")

    def log_step(self, step: AgentStep) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": "step",
            "run_id": self.run_id,
            "timestamp": step.timestamp,
            "step": step.step,
            "thought": step.thought,
            "tool_call": step.tool_call.model_dump() if step.tool_call else None,
            "tool_result": step.tool_result.model_dump() if step.tool_result else None,
            "duration_ms": round(step.duration_ms, 3),
            "diff": step.diff,
            "step_tokens": step.step_tokens.model_dump(),
            "cumulative_tokens": step.cumulative_tokens,
        }
        self._emit(record)
        return record

    def log_summary(
        self,
        *,
        task: str,
        workspace: str,
        stop_reason: StopReason,
        steps_taken: int,
        total_usage: TokenUsage,
        total_duration_ms: float,
        final_answer: str,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": "run_complete",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "workspace": workspace,
            "stop_reason": stop_reason.value,
            "steps_taken": steps_taken,
            "total_tokens": total_usage.total_tokens,
            "total_usage": total_usage.model_dump(),
            "total_duration_ms": round(total_duration_ms, 3),
            "final_answer": final_answer,
        }
        self._emit(record)
        return record

    def _emit(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.trace_path is not None:
            with self.trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(trace_path: Path | str) -> list[dict[str, Any]]:
        path = Path(trace_path)
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records
