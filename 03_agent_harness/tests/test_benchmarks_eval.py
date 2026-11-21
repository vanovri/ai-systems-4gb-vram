"""End-to-end evaluation tests across all 3 real buggy benchmark repositories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.cli import main as cli_main
from harness.eval import evaluate_benchmark, run_eval_suite
from harness.trace import TraceLogger


@pytest.mark.parametrize("benchmark_name", ["checkout_cart", "rate_limiter", "csv_parser"])
def test_agent_solves_each_benchmark_end_to_end(benchmark_name: str, tmp_path: Path) -> None:
    record = evaluate_benchmark(benchmark_name, trace_dir=tmp_path)

    assert record.baseline_failed is True, f"{benchmark_name} must start with failing tests"
    assert record.post_fix_passed is True, f"{benchmark_name} tests must pass after agent fix"
    assert record.solved is True
    assert record.stop_reason == "completed"
    assert record.diff_present is True
    assert record.trace_path is not None

    trace_records = TraceLogger.load_jsonl(record.trace_path)
    tool_names = [r["tool_call"]["name"] for r in trace_records if r.get("tool_call")]
    assert "list_files" in tool_names
    assert "read_file" in tool_names
    assert "edit_file" in tool_names
    assert tool_names.count("run_tests") >= 2


def test_cli_eval_writes_summary_and_traces(tmp_path: Path) -> None:
    exit_code = cli_main(["eval", "--provider", "offline", "--out-dir", str(tmp_path)])
    assert exit_code == 0

    summary_path = tmp_path / "eval_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total"] == 3
    assert summary["solved"] == 3
    assert summary["pass_rate"] == 1.0
    for name in ("checkout_cart", "rate_limiter", "csv_parser"):
        assert (tmp_path / f"{name}.jsonl").exists()


def test_run_eval_suite_leaves_source_benchmarks_buggy_and_intact(tmp_path: Path) -> None:
    records = run_eval_suite(trace_dir=tmp_path)
    assert all(r.solved for r in records)
    # Re-running confirms original benchmark directories were never mutated in place
    second_run = evaluate_benchmark("checkout_cart", trace_dir=tmp_path)
    assert second_run.baseline_failed is True
    assert second_run.solved is True
