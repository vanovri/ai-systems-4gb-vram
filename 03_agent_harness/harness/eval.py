"""Automated benchmark evaluation suite for AgentHarness."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from harness.agent import AgentConfig, AgentHarness
from harness.llm import HeuristicOfflineProvider, LLMProvider
from harness.tools import ToolRegistry

BENCHMARK_TASKS: dict[str, str] = {
    "checkout_cart": (
        "Investigate the failing unit tests in this checkout cart repository, "
        "fix the bug in cart.py, and verify all tests pass."
    ),
    "rate_limiter": (
        "Investigate the failing unit tests in this sliding-window rate limiter repository, "
        "fix the off-by-one bug in limiter.py, and verify all tests pass."
    ),
    "csv_parser": (
        "Investigate the failing unit tests in this CSV parser repository, "
        "fix the quoted-comma and escaped-quote parsing bug in parser.py, and verify all tests pass."
    ),
}


@dataclass(slots=True)
class BenchmarkEvalRecord:
    benchmark: str
    baseline_failed: bool
    post_fix_passed: bool
    solved: bool
    stop_reason: str
    steps_taken: int
    total_tokens: int
    total_duration_ms: float
    trace_path: str | None
    diff_present: bool


def default_benchmarks_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmarks"


def evaluate_benchmark(
    benchmark_name: str,
    *,
    benchmarks_dir: Path | None = None,
    provider_factory: Callable[[], LLMProvider] = HeuristicOfflineProvider,
    trace_dir: Path | None = None,
    config: AgentConfig | None = None,
) -> BenchmarkEvalRecord:
    base_dir = benchmarks_dir or default_benchmarks_dir()
    source_repo = base_dir / benchmark_name
    if not source_repo.is_dir():
        raise FileNotFoundError(f"Benchmark repo not found: {source_repo}")

    task = BENCHMARK_TASKS.get(
        benchmark_name,
        f"Fix the failing unit tests in benchmark '{benchmark_name}' and confirm pytest passes.",
    )

    with tempfile.TemporaryDirectory(prefix=f"harness_{benchmark_name}_") as tmp_dir:
        work_repo = Path(tmp_dir) / benchmark_name
        shutil.copytree(source_repo, work_repo)

        tools = ToolRegistry(work_repo)
        baseline = tools.run_tests("pytest -q")
        baseline_failed = not baseline.ok

        trace_path = (trace_dir / f"{benchmark_name}.jsonl") if trace_dir else None
        harness = AgentHarness(
            provider=provider_factory(),
            tools=tools,
            config=config or AgentConfig(max_steps=12),
        )
        result = harness.run(task, trace_path=trace_path, run_id=f"eval-{benchmark_name}")

        post_check = tools.run_tests("pytest -q")
        post_fix_passed = post_check.ok
        diff_present = any(step.diff for step in result.steps)
        solved = baseline_failed and post_fix_passed and result.stop_reason.value == "completed"

        return BenchmarkEvalRecord(
            benchmark=benchmark_name,
            baseline_failed=baseline_failed,
            post_fix_passed=post_fix_passed,
            solved=solved,
            stop_reason=result.stop_reason.value,
            steps_taken=result.steps_taken,
            total_tokens=result.total_usage.total_tokens,
            total_duration_ms=result.total_duration_ms,
            trace_path=str(trace_path) if trace_path else None,
            diff_present=diff_present,
        )


def run_eval_suite(
    *,
    benchmarks: list[str] | None = None,
    benchmarks_dir: Path | None = None,
    provider_factory: Callable[[], LLMProvider] = HeuristicOfflineProvider,
    trace_dir: Path | str | None = None,
    config: AgentConfig | None = None,
) -> list[BenchmarkEvalRecord]:
    selected = benchmarks or list(BENCHMARK_TASKS.keys())
    out_dir = Path(trace_dir) if trace_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    records: list[BenchmarkEvalRecord] = []
    for name in selected:
        rec = evaluate_benchmark(
            name,
            benchmarks_dir=benchmarks_dir,
            provider_factory=provider_factory,
            trace_dir=out_dir,
            config=config,
        )
        records.append(rec)

    if out_dir is not None:
        summary_path = out_dir / "eval_summary.json"
        summary_payload = {
            "total": len(records),
            "solved": sum(1 for r in records if r.solved),
            "pass_rate": round(sum(1 for r in records if r.solved) / max(1, len(records)), 4),
            "benchmarks": [asdict(r) for r in records],
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return records
