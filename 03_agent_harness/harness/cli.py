"""Command-line interface for running the agent harness or its benchmark eval suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness.agent import AgentConfig, AgentHarness
from harness.eval import BENCHMARK_TASKS, run_eval_suite
from harness.llm import HeuristicOfflineProvider, LLMProvider, OpenAICompatibleProvider
from harness.tools import ToolRegistry


def _make_provider(provider_name: str, model: str | None = None, base_url: str | None = None) -> LLMProvider:
    if provider_name == "offline":
        return HeuristicOfflineProvider()
    if provider_name == "openai":
        return OpenAICompatibleProvider(model=model, base_url=base_url)
    raise ValueError(f"Unsupported provider: {provider_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness.cli",
        description="Minimal coding agent harness with 4 sandboxed tools and bounded control loop.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run the agent harness against a target workspace.")
    run_cmd.add_argument("--workspace", required=True, help="Path to target workspace directory.")
    run_cmd.add_argument("--task", required=True, help="Task prompt for the agent.")
    run_cmd.add_argument(
        "--provider",
        choices=["offline", "openai"],
        default="offline",
        help="LLM backend (offline heuristic or openai-compatible endpoint).",
    )
    run_cmd.add_argument("--model", default=None, help="Override OPENAI_MODEL for openai provider.")
    run_cmd.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL for openai provider.")
    run_cmd.add_argument("--max-steps", type=int, default=12, help="Maximum bounded loop steps (Tmax).")
    run_cmd.add_argument("--max-tokens", type=int, default=64000, help="Cumulative token budget guard.")
    run_cmd.add_argument("--trace", default=None, help="Optional JSONL trace output path.")

    eval_cmd = sub.add_parser("eval", help="Run automated benchmark suite on isolated repo copies.")
    eval_cmd.add_argument(
        "--benchmarks",
        nargs="*",
        default=list(BENCHMARK_TASKS.keys()),
        help="Benchmarks to run (default: all 3).",
    )
    eval_cmd.add_argument(
        "--provider",
        choices=["offline", "openai"],
        default="offline",
        help="LLM backend for benchmark evaluation.",
    )
    eval_cmd.add_argument("--model", default=None, help="Override OPENAI_MODEL for openai provider.")
    eval_cmd.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL for openai provider.")
    eval_cmd.add_argument("--max-steps", type=int, default=12, help="Maximum loop steps per benchmark.")
    eval_cmd.add_argument("--out-dir", default="traces", help="Directory for JSONL traces and eval_summary.json.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        tools = ToolRegistry(Path(args.workspace))
        provider = _make_provider(args.provider, model=args.model, base_url=args.base_url)
        harness = AgentHarness(
            provider=provider,
            tools=tools,
            config=AgentConfig(max_steps=args.max_steps, max_token_budget=args.max_tokens),
        )
        result = harness.run(args.task, trace_path=args.trace)
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "stop_reason": result.stop_reason.value,
                    "steps_taken": result.steps_taken,
                    "total_tokens": result.total_usage.total_tokens,
                    "total_duration_ms": result.total_duration_ms,
                    "trace_path": result.trace_path,
                    "final_answer": result.final_answer,
                },
                indent=2,
            )
        )
        return 0 if result.stop_reason.value == "completed" else 1

    if args.command == "eval":
        records = run_eval_suite(
            benchmarks=args.benchmarks,
            provider_factory=lambda: _make_provider(
                args.provider, model=args.model, base_url=args.base_url
            ),
            trace_dir=Path(args.out_dir),
            config=AgentConfig(max_steps=args.max_steps),
        )
        solved_count = sum(1 for r in records if r.solved)
        print(f"Benchmark Evaluation: {solved_count}/{len(records)} solved")
        for rec in records:
            status = "PASS" if rec.solved else "FAIL"
            print(
                f"  [{status}] {rec.benchmark:15s} | baseline_fail={rec.baseline_failed} "
                f"| post_pass={rec.post_fix_passed} | steps={rec.steps_taken} "
                f"| stop={rec.stop_reason} | trace={rec.trace_path}"
            )
        return 0 if solved_count == len(records) else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
