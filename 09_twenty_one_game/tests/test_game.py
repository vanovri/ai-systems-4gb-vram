"""Comprehensive test suite for Project #9 — 21 Number Game (Bachet-Nim)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from game.ai import (
    choose_move,
    greedy_opponent_move,
    optimal_modulo_move,
    solve_minimax,
    verify_unbeatable_against_minimax,
)
from game.cli import format_replay_log, main as cli_main, render_progress_bar, run_auto_demo, run_interactive_session
from game.engine import GameEngine, InvalidMoveError
from game.simulation import run_benchmark_suite, simulate_matches


def test_single_shared_counter_fixes_slide_bug() -> None:
    """Players must advance one single shared sequence 1..21 rather than separate counters."""
    s0 = GameEngine.new_game(first_player="human", mode="misere")
    s1 = GameEngine.apply_move(s0, 2)  # Human takes 1, 2
    s2 = GameEngine.apply_move(s1, 2)  # AI takes 3, 4 (continuing from 2, NOT restarting at 0)

    assert s0.current == 0  # Immutability check
    assert s1.current == 2
    assert s1.move_history[0].numbers == (1, 2)
    assert s2.current == 4
    assert s2.move_history[1].numbers == (3, 4)
    assert s2.next_player == "human"


def test_strict_move_validation_and_bounds() -> None:
    state = GameEngine.new_game(first_player="human", mode="misere")
    for bad in (-1, 0, 4, 21, True, "2"):
        with pytest.raises(InvalidMoveError):
            GameEngine.apply_move(state, bad)  # type: ignore[arg-type]

    # Near target=21: at current=19, only 1 and 2 are legal (3 would exceed 21)
    near_end = state
    for step in (3, 3, 3, 3, 3, 3, 1):
        near_end = GameEngine.apply_move(near_end, step)
    assert near_end.current == 19
    assert GameEngine.legal_moves(near_end) == (1, 2)
    with pytest.raises(InvalidMoveError):
        GameEngine.apply_move(near_end, 3)


def test_misere_mode_taking_21_loses() -> None:
    state = GameEngine.new_game(first_player="human", mode="misere")
    # Sequence of moves reaching 20 cleanly: 2+2=4, 3+1=8, 1+3=12, 2+2=16, 3+1=20 -> human forced to take 21
    for step in (2, 2, 3, 1, 1, 3, 2, 2, 3, 1):
        state = GameEngine.apply_move(state, step)
    assert state.current == 20
    assert state.next_player == "human"
    assert GameEngine.legal_moves(state) == (1,)

    final = GameEngine.apply_move(state, 1)
    assert final.is_over is True
    assert final.current == 21
    assert final.loser == "human"
    assert final.winner == "ai"

    with pytest.raises(InvalidMoveError, match="already finished"):
        GameEngine.apply_move(final, 1)


def test_normal_mode_taking_21_wins() -> None:
    state = GameEngine.new_game(first_player="ai", mode="normal")
    # AI plays modulo-4 normal strategy: 1, 5, 9, 13, 17, 21
    rng = random.Random(99)
    while not state.is_over:
        if state.next_player == "ai":
            move = choose_move(state, difficulty="unbeatable", rng=rng)
        else:
            move = rng.choice(GameEngine.legal_moves(state))
        state = GameEngine.apply_move(state, move)
    assert state.current == 21
    assert state.winner == "ai"
    assert state.loser == "human"


def test_exhaustive_minimax_proof_misere() -> None:
    """Exhaustive game-tree proof across all 21 states in misere mode."""
    proof = verify_unbeatable_against_minimax(target=21, max_step=3, mode="misere")
    assert proof["states_checked"] == 21
    assert proof["proven_optimal"] is True
    assert proof["mismatches"] == []

    # Verify exact P-positions (losing states for player to move) are multiples of 4: 0, 4, 8, 12, 16, 20
    losing = [c for c in range(21) if not solve_minimax(c, 21, 3, "misere")[0]]
    assert losing == [0, 4, 8, 12, 16, 20]


def test_exhaustive_minimax_proof_normal() -> None:
    """Exhaustive game-tree proof across all 21 states in normal mode."""
    proof = verify_unbeatable_against_minimax(target=21, max_step=3, mode="normal")
    assert proof["states_checked"] == 21
    assert proof["proven_optimal"] is True
    assert proof["mismatches"] == []

    # Verify exact P-positions in normal mode are 1, 5, 9, 13, 17
    losing = [c for c in range(21) if not solve_minimax(c, 21, 3, "normal")[0]]
    assert losing == [1, 5, 9, 13, 17]


def test_modulo_4_targets_safe_multiples() -> None:
    for c in range(20):
        opt = optimal_modulo_move(c, target=21, max_step=3, mode="misere")
        if c % 4 == 0:
            assert opt is None
        else:
            assert opt is not None
            assert 1 <= opt <= 3
            assert (c + opt) % 4 == 0


def test_ai_difficulties_always_return_legal_moves() -> None:
    rng = random.Random(123)
    for diff in ("easy", "smart", "unbeatable"):
        for mode in ("misere", "normal"):
            state = GameEngine.new_game(first_player="human", mode=mode)  # type: ignore[arg-type]
            while not state.is_over:
                move = choose_move(state, difficulty=diff, rng=rng)  # type: ignore[arg-type]
                assert move in GameEngine.legal_moves(state)
                state = GameEngine.apply_move(state, move)
            assert state.is_over


def test_smart_ai_plays_optimally_in_endgame() -> None:
    # From c=17, 18, 19 in misere, smart AI must hit 20 to trap opponent on 21
    rng = random.Random(0)
    for c in (17, 18, 19):
        state = GameEngine.new_game(first_player="ai", mode="misere")
        # Construct state cleanly via legal moves
        remaining = c
        while remaining > 0:
            step = min(3, remaining)
            state = GameEngine.apply_move(state, step)
            remaining -= step
        move = choose_move(state, difficulty="smart", rng=rng)
        assert state.current + move == 20


def test_simulation_1000_matches_unbeatable_second_vs_random_100_percent() -> None:
    res = simulate_matches(
        matches=1000,
        ai_difficulty="unbeatable",
        opponent_policy="random",
        first_player="human",
        mode="misere",
        seed=42,
    )
    assert res["matches"] == 1000
    assert res["ai_wins"] == 1000
    assert res["opponent_wins"] == 0
    assert res["ai_win_rate_pct"] == 100.0


def test_simulation_1000_matches_unbeatable_second_vs_greedy_100_percent() -> None:
    res = simulate_matches(
        matches=1000,
        ai_difficulty="unbeatable",
        opponent_policy="greedy",
        first_player="human",
        mode="misere",
        seed=43,
    )
    assert res["matches"] == 1000
    assert res["ai_wins"] == 1000
    assert res["opponent_wins"] == 0
    assert res["ai_win_rate_pct"] == 100.0


def test_simulation_1000_matches_normal_unbeatable_first_100_percent() -> None:
    res = simulate_matches(
        matches=1000,
        ai_difficulty="unbeatable",
        opponent_policy="mixed",
        first_player="ai",
        mode="normal",
        seed=44,
    )
    assert res["ai_wins"] == 1000
    assert res["ai_win_rate_pct"] == 100.0


def test_cli_auto_demo_progress_bar_and_replay_log() -> None:
    bar = render_progress_bar(10, 21)
    assert "10/21" in bar
    assert bar.startswith("[") and "]" in bar

    final_state = run_auto_demo(mode="misere", difficulty="unbeatable", first_player="human", seed=7)
    assert final_state.is_over is True
    assert final_state.winner == "ai"
    replay = format_replay_log(final_state)
    assert "Match Replay Log" in replay
    assert "winner=ai" in replay


def test_cli_interactive_and_simulate_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Interactive session with scripted inputs (human plays 1 repeatedly against unbeatable AI, then declines replay)
    inputs = iter(["invalid", "1", "1", "1", "1", "1", "1", "n"])
    outputs: list[str] = []
    board = run_interactive_session(
        mode="misere",
        difficulty="unbeatable",
        first_player="human",
        input_fn=lambda _: next(inputs),
        output_fn=outputs.append,
    )
    assert board["games"] == 1
    assert board["ai"] == 1
    assert board["human"] == 0

    out_json = tmp_path / "sim.json"
    rc = cli_main(["--simulate", "100", "--output", str(out_json)])
    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["minimax_proof"]["misere"]["proven_optimal"] is True
    assert data["simulations"]["misere_unbeatable_second_vs_random"]["ai_win_rate_pct"] == 100.0
    capsys.readouterr()
