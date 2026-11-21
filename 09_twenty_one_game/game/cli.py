"""Interactive TUI & non-interactive demo/simulation CLI for 21 Number Game."""

from __future__ import annotations

import argparse
import json
import random
from typing import Callable, Sequence

from game.ai import Difficulty, choose_move
from game.engine import GameEngine, GameMode, GameState, InvalidMoveError, Player
from game.simulation import run_benchmark_suite


def render_progress_bar(current: int, target: int = 21, width: int = 21) -> str:
    """Render an ASCII progress bar representing shared counter progress 0 -> 21."""
    clamped = max(0, min(current, target))
    filled = round((clamped / target) * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {clamped:2d}/{target}"


def format_replay_log(state: GameState) -> str:
    """Format full turn-by-turn replay log of a match."""
    lines = ["--- Match Replay Log ---"]
    for rec in state.move_history:
        actor = "Human" if rec.player == "human" else "AI   "
        nums = ", ".join(str(n) for n in rec.numbers)
        lines.append(
            f"Turn {rec.turn:2d} | {actor} took {rec.count} ({nums:<10}) -> "
            f"{render_progress_bar(rec.new_total, state.target)}"
        )
    lines.append(f"Result: winner={state.winner}, loser={state.loser} (mode={state.mode})")
    return "\n".join(lines)


def run_auto_demo(
    *,
    mode: GameMode = "misere",
    difficulty: Difficulty = "unbeatable",
    first_player: Player = "human",
    seed: int = 7,
) -> GameState:
    """Execute a complete non-interactive showcase match with ASCII progress and replay log."""
    rng = random.Random(seed)
    state = GameEngine.new_game(first_player=first_player, mode=mode)

    print("=== 21 Number Game (Bachet-Nim) — Auto Demo ===")
    print(f"Rules: Shared counter 1..{state.target} | Mode: {mode} | AI: {difficulty} | First: {first_player}")
    print(f"Start: {render_progress_bar(state.current, state.target)}\n")

    while not state.is_over:
        if state.next_player == "human":
            move = rng.choice(GameEngine.legal_moves(state))
            actor = "Demo-Player"
        else:
            move = choose_move(state, difficulty=difficulty, rng=rng)
            actor = f"AI ({difficulty})"
        state = GameEngine.apply_move(state, move)
        last = state.move_history[-1]
        nums = ", ".join(str(n) for n in last.numbers)
        print(
            f"Turn {last.turn:2d} | {actor:<16} picks +{last.count} [{nums:<10}] "
            f"{render_progress_bar(state.current, state.target)}"
        )

    print()
    print(format_replay_log(state))
    return state


def run_interactive_session(
    *,
    mode: GameMode = "misere",
    difficulty: Difficulty = "unbeatable",
    first_player: Player = "human",
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, int]:
    """Run an interactive multi-round TUI session with scoreboard and replay logs."""
    scoreboard = {"human": 0, "ai": 0, "games": 0}
    rng = random.Random()

    output_fn("=== 21 Number Game (Classic Bachet-Nim) ===")
    rule_line = (
        "Whoever is forced to say 21 LOSES."
        if mode == "misere"
        else "Whoever says 21 WINS."
    )
    output_fn(f"Mode: {mode} ({rule_line}) | Difficulty: {difficulty} | First: {first_player}")

    while True:
        state = GameEngine.new_game(first_player=first_player, mode=mode)
        output_fn(f"\nNew Match | {render_progress_bar(state.current, state.target)}")

        while not state.is_over:
            valid = GameEngine.legal_moves(state)
            if state.next_player == "human":
                while True:
                    raw = input_fn(f"Your turn — pick {list(valid)} numbers (or 'q' to quit): ").strip().lower()
                    if raw in {"q", "quit"}:
                        output_fn(f"Session ended. Scoreboard: {scoreboard}")
                        return scoreboard
                    try:
                        step = int(raw)
                        state = GameEngine.apply_move(state, step)
                        break
                    except (ValueError, InvalidMoveError) as exc:
                        output_fn(f"Invalid input: {exc}")
                last = state.move_history[-1]
                output_fn(
                    f"You picked {list(last.numbers)} -> {render_progress_bar(state.current, state.target)}"
                )
            else:
                step = choose_move(state, difficulty=difficulty, rng=rng)
                state = GameEngine.apply_move(state, step)
                last = state.move_history[-1]
                output_fn(
                    f"AI ({difficulty}) picked {list(last.numbers)} -> "
                    f"{render_progress_bar(state.current, state.target)}"
                )

        scoreboard["games"] += 1
        assert state.winner is not None
        scoreboard[state.winner] += 1
        output_fn("\n" + format_replay_log(state))
        output_fn(
            f"Session Scoreboard -> Human: {scoreboard['human']} | "
            f"AI: {scoreboard['ai']} | Games: {scoreboard['games']}"
        )

        again = input_fn("Play again? [y/N]: ").strip().lower()
        if again not in {"y", "yes"}:
            break

    return scoreboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twenty-one-game",
        description="21 Number Game (Bachet-Nim) with provably optimal modulo-4 AI.",
    )
    parser.add_argument("--mode", choices=["misere", "normal"], default="misere")
    parser.add_argument("--difficulty", choices=["easy", "smart", "unbeatable"], default="unbeatable")
    parser.add_argument("--first", choices=["human", "ai"], default="human")
    parser.add_argument(
        "--auto-demo",
        action="store_true",
        help="Run a complete non-interactive demo match with ASCII progress bar and replay log.",
    )
    parser.add_argument(
        "--simulate",
        type=int,
        default=0,
        help="Run N-match benchmark simulation (e.g. --simulate 1000) and minimax proof.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path to save simulation benchmark report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.simulate > 0:
        report = run_benchmark_suite(
            matches=args.simulate,
            seed=args.seed,
            output_path=args.output,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.auto_demo:
        run_auto_demo(
            mode=args.mode,
            difficulty=args.difficulty,
            first_player=args.first,
            seed=args.seed,
        )
        return 0

    run_interactive_session(
        mode=args.mode,
        difficulty=args.difficulty,
        first_player=args.first,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
