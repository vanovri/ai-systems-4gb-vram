"""Structured execution trace writer for multi-agent runs (execution_trace.json)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    """One agent task execution recorded in execution_trace.json."""

    step_index: int
    agent_role: str
    task_name: str
    revision_round: int = 0
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    tools_called: list[str] = Field(default_factory=list)
    handoff_schema: str
    handoff_valid: bool = True
    handoff_summary: str = ""
    handoff_payload: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Full structured execution trace saved to execution_trace.json."""

    run_id: str
    topic: str
    process: str
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    total_duration_ms: float = 0.0
    total_steps: int = 0
    revision_loop_count: int = 0
    quality_threshold: float = 0.85
    quality_scores: list[float] = Field(default_factory=list)
    final_quality_score: float | None = None
    passed_quality_gate: bool = False
    steps: list[TraceStep] = Field(default_factory=list)

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out
