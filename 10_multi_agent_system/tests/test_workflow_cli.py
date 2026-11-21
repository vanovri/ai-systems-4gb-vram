"""Integration tests for execution_trace.json persistence, citation validity, and CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

from multi_agent.cli import main as cli_main
from multi_agent.workflow import run_research_workflow


def test_workflow_writes_execution_trace_json_with_timings_and_handoffs(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    trace_path = tmp_path / "execution_trace.json"

    result = run_research_workflow(
        "Enterprise Multi-Agent Verification Pipelines",
        report_path=report_path,
        trace_path=trace_path,
        simulate_first_draft_gaps=True,
    )

    assert report_path.exists()
    assert trace_path.exists()

    trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_data["total_steps"] == 6
    assert trace_data["revision_loop_count"] == 1
    assert trace_data["passed_quality_gate"] is True
    assert trace_data["total_duration_ms"] >= 0.0
    assert len(trace_data["steps"]) == 6
    roles = [s["agent_role"] for s in trace_data["steps"]]
    assert roles == [
        "Researcher",
        "Analyst",
        "ReportWriter",
        "CriticVerifier",
        "ReportWriter",
        "CriticVerifier",
    ]
    assert all(s["handoff_valid"] is True for s in trace_data["steps"])
    assert result.report_markdown == report_path.read_text(encoding="utf-8")


def test_final_report_citations_all_resolve_to_researcher_facts(tmp_path: Path) -> None:
    report_path = tmp_path / "cited.md"
    result = run_research_workflow(
        "Evaluating Multi-Agent AI Architectures",
        report_path=report_path,
    )

    research_facts = result.task_outputs["research_gathering"]["facts"]
    valid_ids = {f["id"] for f in research_facts}
    cited_ids = set(re.findall(r"\[(S\d+)\]", result.report_markdown))

    assert len(cited_ids) >= 3
    assert cited_ids.issubset(valid_ids)


def test_cli_end_to_end_sequential_and_hierarchical(tmp_path: Path, capsys) -> None:
    seq_report = tmp_path / "seq_report.md"
    seq_trace = tmp_path / "seq_trace.json"
    rc_seq = cli_main(
        [
            "Evaluating Multi-Agent AI Architectures",
            "--process",
            "sequential",
            "--out",
            str(seq_report),
            "--trace",
            str(seq_trace),
            "--json-summary",
        ]
    )
    assert rc_seq == 0
    seq_summary = json.loads(capsys.readouterr().out)
    assert seq_summary["revision_loop_count"] == 1
    assert seq_summary["passed_quality_gate"] is True
    assert seq_report.exists() and seq_trace.exists()

    hier_report = tmp_path / "hier_report.md"
    hier_trace = tmp_path / "hier_trace.json"
    rc_hier = cli_main(
        [
            "Evaluating Multi-Agent AI Architectures",
            "--process",
            "hierarchical",
            "--out",
            str(hier_report),
            "--trace",
            str(hier_trace),
            "--json-summary",
        ]
    )
    assert rc_hier == 0
    hier_summary = json.loads(capsys.readouterr().out)
    assert hier_summary["process"] == "hierarchical"
    assert hier_summary["total_steps"] == 7  # Manager + 4 initial + 1 revision + 1 re-audit
    assert hier_summary["passed_quality_gate"] is True
