"""21 Number Game (Bachet-Nim) pure state-machine engine, optimal AI, and simulator."""

from game.ai import (
    Difficulty,
    choose_move,
    greedy_opponent_move,
    solve_minimax,
    verify_unbeatable_against_minimax,
)
from game.engine import (
    GameEngine,
    GameMode,
    GameState,
    InvalidMoveError,
    MoveRecord,
    Player,
)
from game.simulation import run_benchmark_suite, simulate_matches

__all__ = [
    "Difficulty",
    "GameEngine",
    "GameMode",
    "GameState",
    "InvalidMoveError",
    "MoveRecord",
    "Player",
    "choose_move",
    "greedy_opponent_move",
    "run_benchmark_suite",
    "simulate_matches",
    "solve_minimax",
    "verify_unbeatable_against_minimax",
]
