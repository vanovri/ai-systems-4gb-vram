"""AI policies (easy, smart, unbeatable modulo-4) and exhaustive minimax proof solver."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any, Literal

from game.engine import GameEngine, GameMode, GameState

Difficulty = Literal["easy", "smart", "unbeatable"]


def optimal_modulo_move(
    current: int,
    *,
    target: int = 21,
    max_step: int = 3,
    mode: GameMode = "misere",
) -> int | None:
    """Compute the exact winning modulo-(max_step+1) move, or None if in a P-position.

    Classic 21 (target=21, max_step=3, cycle=4):
    - misere (taking 21 loses):
        Safe targets for opponent are 20, 16, 12, 8, 4 (all multiples of 4).
        Winning step is (target - 1 - current) % (max_step + 1).
    - normal (taking 21 wins):
        Safe targets for opponent are 21, 17, 13, 9, 5, 1.
        Winning step is (target - current) % (max_step + 1).
    """
    if current >= target:
        return None
    cycle = max_step + 1
    goal = (target - 1) if mode == "misere" else target
    remainder = (goal - current) % cycle
    if remainder == 0:
        return None
    if current + remainder <= target:
        return remainder
    return None


def choose_move(
    state: GameState,
    difficulty: Difficulty = "unbeatable",
    rng: random.Random | None = None,
) -> int:
    """Select a legal move according to the requested AI difficulty."""
    valid = GameEngine.legal_moves(state)
    if not valid:
        raise ValueError("Cannot choose a move on a finished game state.")

    gen = rng if rng is not None else random.Random()

    if difficulty == "easy":
        return gen.choice(valid)

    if difficulty == "smart":
        # Tactical endgame awareness at >= 14: lock onto optimal modulo-4 winning track
        endgame_threshold = max(0, state.target - 7)
        if state.current >= endgame_threshold:
            opt = optimal_modulo_move(
                state.current,
                target=state.target,
                max_step=state.max_step,
                mode=state.mode,
            )
            if opt is not None and opt in valid:
                return opt
            # Avoid voluntarily hitting target in misere if safer move exists
            if state.mode == "misere" and len(valid) > 1:
                safe = [m for m in valid if state.current + m < state.target]
                if safe:
                    return min(safe)
            return valid[0]

        # Early/mid-game: mix occasional optimal play with exploratory moves
        if gen.random() < 0.35:
            opt = optimal_modulo_move(
                state.current,
                target=state.target,
                max_step=state.max_step,
                mode=state.mode,
            )
            if opt is not None and opt in valid:
                return opt
        return gen.choice(valid)

    if difficulty == "unbeatable":
        opt = optimal_modulo_move(
            state.current,
            target=state.target,
            max_step=state.max_step,
            mode=state.mode,
        )
        if opt is not None and opt in valid:
            return opt
        # In a theoretical losing state (e.g. moving first in misere at 0),
        # take 1 to maximize remaining turns where a human/suboptimal opponent may err.
        return valid[0]

    raise ValueError(f"Unknown difficulty '{difficulty}'. Expected 'easy', 'smart', or 'unbeatable'.")


def greedy_opponent_move(state: GameState) -> int:
    """Deterministic greedy opponent for simulation benchmarks.

    Takes the largest safe move (max_step) without voluntarily taking 21 in misere,
    or grabs 21 immediately in normal mode when reachable.
    """
    valid = GameEngine.legal_moves(state)
    if not valid:
        raise ValueError("No legal moves.")
    if state.mode == "normal":
        if state.target - state.current in valid:
            return state.target - state.current
        return valid[-1]
    # Misere: take largest step that does not immediately force self onto target
    safe = [m for m in valid if state.current + m < state.target]
    return safe[-1] if safe else valid[0]


@lru_cache(maxsize=None)
def solve_minimax(
    current: int,
    target: int = 21,
    max_step: int = 3,
    mode: GameMode = "misere",
) -> tuple[bool, tuple[int, ...]]:
    """Exhaustive game-tree minimax solver for 21 Bachet-Nim.

    Returns:
        (can_current_player_force_win, tuple_of_all_winning_moves)
    """
    if current >= target:
        raise ValueError("solve_minimax expects a non-terminal state (current < target).")

    upper = min(max_step, target - current)
    winning_moves: list[int] = []

    for step in range(1, upper + 1):
        nxt = current + step
        if nxt == target:
            move_wins = mode == "normal"
        else:
            opp_can_win, _ = solve_minimax(nxt, target, max_step, mode)
            move_wins = not opp_can_win
        if move_wins:
            winning_moves.append(step)

    return (len(winning_moves) > 0, tuple(winning_moves))


def verify_unbeatable_against_minimax(
    *,
    target: int = 21,
    max_step: int = 3,
    mode: GameMode = "misere",
) -> dict[str, Any]:
    """Exhaustively prove unbeatable modulo-4 policy matches minimax across all states 0..target-1."""
    checked = 0
    winning_states = 0
    losing_states = 0
    mismatches: list[dict[str, Any]] = []

    for current in range(target):
        checked += 1
        can_win, minimax_winners = solve_minimax(current, target, max_step, mode)
        state = GameState(
            target=target,
            max_step=max_step,
            mode=mode,
            current=current,
            next_player="ai",
        )
        ai_move = choose_move(state, difficulty="unbeatable")

        if can_win:
            winning_states += 1
            if ai_move not in minimax_winners:
                mismatches.append(
                    {
                        "current": current,
                        "ai_move": ai_move,
                        "minimax_winners": list(minimax_winners),
                    }
                )
        else:
            losing_states += 1

    return {
        "mode": mode,
        "target": target,
        "max_step": max_step,
        "states_checked": checked,
        "winning_states": winning_states,
        "losing_states": losing_states,
        "mismatches": mismatches,
        "proven_optimal": len(mismatches) == 0,
    }
