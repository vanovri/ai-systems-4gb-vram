"""Pure immutable state-machine engine for the classic 21 Number Game (Bachet-Nim).

Fixes the common slide bug of tracking two separate counters (`player += p`, `computer += r`):
players share ONE single sequence of consecutive numbers from 1 to 21.
On each turn a player takes 1, 2, or 3 consecutive numbers.
- misere (classic 21): whoever is forced to take 21 LOSES.
- normal (variant): whoever takes 21 WINS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GameMode = Literal["misere", "normal"]
Player = Literal["human", "ai"]


class InvalidMoveError(ValueError):
    """Raised when a move violates 21 Number Game rules."""


@dataclass(frozen=True)
class MoveRecord:
    """Immutable record of a single turn in the shared sequence."""

    turn: int
    player: Player
    count: int
    numbers: tuple[int, ...]
    new_total: int


@dataclass(frozen=True)
class GameState:
    """Immutable game snapshot."""

    target: int = 21
    max_step: int = 3
    mode: GameMode = "misere"
    current: int = 0
    next_player: Player = "human"
    move_history: tuple[MoveRecord, ...] = ()
    winner: Player | None = None
    loser: Player | None = None

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.current)


def opponent_of(player: Player) -> Player:
    return "ai" if player == "human" else "human"


class GameEngine:
    """Stateless pure transition engine for 21 Number Game."""

    @staticmethod
    def new_game(
        *,
        first_player: Player = "human",
        mode: GameMode = "misere",
        target: int = 21,
        max_step: int = 3,
    ) -> GameState:
        if first_player not in ("human", "ai"):
            raise ValueError(f"Invalid first_player '{first_player}'. Expected 'human' or 'ai'.")
        if mode not in ("misere", "normal"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'misere' or 'normal'.")
        if target < 2:
            raise ValueError("target must be >= 2")
        if max_step < 1 or max_step >= target:
            raise ValueError("max_step must satisfy 1 <= max_step < target")

        return GameState(
            target=target,
            max_step=max_step,
            mode=mode,
            current=0,
            next_player=first_player,
            move_history=(),
            winner=None,
            loser=None,
        )

    @staticmethod
    def legal_moves(state: GameState) -> tuple[int, ...]:
        """Return valid counts in 1..min(max_step, remaining) or empty tuple if finished."""
        if state.is_over or state.current >= state.target:
            return ()
        upper = min(state.max_step, state.target - state.current)
        return tuple(range(1, upper + 1))

    @staticmethod
    def apply_move(state: GameState, count: int) -> GameState:
        """Apply a strictly validated move and return a brand-new immutable GameState."""
        if state.is_over:
            raise InvalidMoveError("Game is already finished.")
        if not isinstance(count, int) or isinstance(count, bool):
            raise InvalidMoveError("Move count must be an integer.")

        valid = GameEngine.legal_moves(state)
        if count not in valid:
            raise InvalidMoveError(
                f"Illegal move {count} at current={state.current} (target={state.target}). "
                f"Allowed moves: {list(valid)}."
            )

        mover = state.next_player
        other = opponent_of(mover)
        numbers = tuple(range(state.current + 1, state.current + count + 1))
        new_total = state.current + count
        record = MoveRecord(
            turn=len(state.move_history) + 1,
            player=mover,
            count=count,
            numbers=numbers,
            new_total=new_total,
        )

        winner: Player | None = None
        loser: Player | None = None
        if new_total == state.target:
            if state.mode == "misere":
                loser = mover
                winner = other
            else:
                winner = mover
                loser = other

        return GameState(
            target=state.target,
            max_step=state.max_step,
            mode=state.mode,
            current=new_total,
            next_player=other,
            move_history=state.move_history + (record,),
            winner=winner,
            loser=loser,
        )
