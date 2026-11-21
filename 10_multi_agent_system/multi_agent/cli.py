"""CLI entrypoint for executing the 4-agent collaborative research & verification crew."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multi_agent.llm import BaseLLM, DeterministicRoleLLM, OpenAICompatibleLLM
from multi_agent.workflow import run_research_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m multi_agent.cli",
        description="CrewAI-style multi-agent system (Researcher, Analyst, ReportWriter, CriticVerifier).",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="Evaluating Multi-Agent AI Architectures for Enterprise Production",
        help="Research and analysis topic.",
    )
    parser.add_argument(
        "--process",
        choices=["sequential", "hierarchical"],
        default="sequential",
        help="Crew execution mode (default: sequential).",
    )
    parser.add_argument(
        "--provider",
        choices=["offline", "openai"],
        default="offline",
        help="LLM backend (offline deterministic or live OpenAI-compatible endpoint).",
    )
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL for openai provider.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL for openai provider.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="CriticVerifier minimum quality score threshold (default: 0.85).",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=1,
        help="Maximum revision loops triggered by CriticVerifier (default: 1).",
    )
    parser.add_argument(
        "--no-gap-sim",
        action="store_true",
        help="Disable initial draft gap simulation in offline mode (draft v1 passes immediately).",
    )
    parser.add_argument(
        "--out",
        default="reports/sample_multi_agent_report.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--trace",
        default="reports/execution_trace.json",
        help="Output execution trace JSON path.",
    )
    parser.add_argument("--json-summary", action="store_true", help="Print run summary as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    llm: BaseLLM
    if args.provider == "openai":
        llm = OpenAICompatibleLLM(model=args.model, base_url=args.base_url)
    else:
        llm = DeterministicRoleLLM(
            simulate_first_draft_gaps=not args.no_gap_sim,
            quality_threshold=args.threshold,
        )

    result = run_research_workflow(
        args.topic,
        llm=llm,
        process=args.process,
        quality_threshold=args.threshold,
        max_revisions=args.max_revisions,
        simulate_first_draft_gaps=not args.no_gap_sim,
        report_path=Path(args.out),
        trace_path=Path(args.trace),
    )

    summary = {
        "run_id": result.run_id,
        "topic": result.topic,
        "process": result.process,
        "total_steps": result.trace.total_steps,
        "revision_loop_count": result.revision_loop_count,
        "quality_scores": result.trace.quality_scores,
        "final_quality_score": result.trace.final_quality_score,
        "passed_quality_gate": result.trace.passed_quality_gate,
        "total_duration_ms": result.trace.total_duration_ms,
        "report_path": str(args.out),
        "trace_path": str(args.trace),
    }

    if args.json_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Multi-Agent Kickoff Complete [{result.process}] | steps={summary['total_steps']} "
            f"| revisions={summary['revision_loop_count']} | scores={summary['quality_scores']} "
            f"| gate_passed={summary['passed_quality_gate']} | report={args.out} | trace={args.trace}"
        )
    return 0 if result.trace.passed_quality_gate else 1


if __name__ == "__main__":
    sys.exit(main())
