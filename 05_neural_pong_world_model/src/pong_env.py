"""Deterministic headless 84x84 binary Pong engine for Neural World Model training.

Design choices specifically optimized for pixel-to-pixel neural world models:
- Pure integer pixel coordinates and integer velocities so the state transition
  is fully observable from 4 consecutive binary 84x84 frames (no hidden sub-pixel
  rounding ambiguity).
- Crisp binary rendering (uint8 values strictly 0 or 255):
  - Top wall: rows 0..1, bottom wall: rows 82..83
  - Dashed center line: cols 41..42 (4 px on, 4 px off)
  - Left AI paddle: cols 5..7, height 14 px
  - Right Player paddle: cols 76..78, height 14 px (controlled by a_t in {0, 1, 2})
  - Ball: 3x3 square, vx in {-2, +2}, vy in {-2, -1, 0, +1, +2}
- Deterministic ball respawn at center (40, 40) after a point is scored within
  an episode, enabling the trained neural world model to learn rally resets too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Canvas & geometry constants
FRAME_H = 84
FRAME_W = 84
WALL_THICKNESS = 2
PLAY_TOP = WALL_THICKNESS  # 2
PLAY_BOTTOM = FRAME_H - WALL_THICKNESS  # 82 (exclusive)

PADDLE_W = 3
PADDLE_H = 14
LEFT_PADDLE_X = 5  # occupies cols 5, 6, 7 -> inner face at col 8
RIGHT_PADDLE_X = 76  # occupies cols 76, 77, 78 -> inner face at col 75
PADDLE_MIN_Y = PLAY_TOP  # 2
PADDLE_MAX_Y = PLAY_BOTTOM - PADDLE_H  # 68
PLAYER_SPEED = 2
AI_SPEED = 2

BALL_SIZE = 3
BALL_MIN_Y = PLAY_TOP  # 2
BALL_MAX_Y = PLAY_BOTTOM - BALL_SIZE  # 79
BALL_SPEED_X = 2

# Actions
ACTION_NONE = 0
ACTION_UP = 1
ACTION_DOWN = 2
NUM_ACTIONS = 3
ACTION_NAMES = {
    ACTION_NONE: "NONE",
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
}


def _build_static_background() -> np.ndarray:
    """Precompute the static 84x84 court background (walls + dashed center line)."""
    bg = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    # Top and bottom walls
    bg[:WALL_THICKNESS, :] = 255
    bg[FRAME_H - WALL_THICKNESS :, :] = 255
    # Dashed center line (cols 41..42, 4 px on / 4 px off)
    for y in range(4, FRAME_H - 4, 8):
        bg[y : min(y + 4, FRAME_H - WALL_THICKNESS), 41:43] = 255
    return bg


STATIC_BACKGROUND: np.ndarray = _build_static_background()


@dataclass
class PongState:
    """Complete Markov state of deterministic Pong."""

    left_y: int
    right_y: int
    ball_x: int
    ball_y: int
    ball_vx: int
    ball_vy: int
    left_score: int = 0
    right_score: int = 0
    step_count: int = 0
    rallies: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PongState":
        return cls(**data)


class PongEnv:
    """Fast deterministic headless Pong environment rendering 84x84 uint8 frames."""

    def __init__(
        self,
        seed: int = 0,
        ai_jitter: float = 0.0,
        max_points: int = 5,
        max_steps: int = 600,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.ai_jitter = float(ai_jitter)
        self.max_points = int(max_points)
        self.max_steps = int(max_steps)
        self.state = PongState(
            left_y=35,
            right_y=35,
            ball_x=40,
            ball_y=40,
            ball_vx=BALL_SPEED_X,
            ball_vy=1,
        )

    def reset(
        self,
        seed: int | None = None,
        state: PongState | None = None,
        randomize_positions: bool = True,
    ) -> np.ndarray:
        """Reset environment to an initial state and return the 84x84 uint8 frame."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if state is not None:
            self.state = PongState(**state.to_dict())
            return self.render()

        if randomize_positions:
            left_y = int(self.rng.integers(PADDLE_MIN_Y + 4, PADDLE_MAX_Y - 3))
            right_y = int(self.rng.integers(PADDLE_MIN_Y + 4, PADDLE_MAX_Y - 3))
            # Start ball across varied court positions to cover bounce & mid-flight states
            ball_x = int(self.rng.integers(20, 64))
            ball_y = int(self.rng.integers(PLAY_TOP + 6, PLAY_BOTTOM - 9))
            ball_vx = int(self.rng.choice([-BALL_SPEED_X, BALL_SPEED_X]))
            ball_vy = int(self.rng.choice([-2, -1, 0, 1, 2]))
        else:
            left_y = 35
            right_y = 35
            ball_x = 40
            ball_y = 40
            ball_vx = BALL_SPEED_X
            ball_vy = 1

        self.state = PongState(
            left_y=left_y,
            right_y=right_y,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_vx=ball_vx,
            ball_vy=ball_vy,
            left_score=0,
            right_score=0,
            step_count=0,
            rallies=0,
        )
        return self.render()

    def get_state(self) -> PongState:
        return PongState(**self.state.to_dict())

    def set_state(self, state: PongState) -> None:
        self.state = PongState(**state.to_dict())

    @staticmethod
    def _bounce_vy(ball_y: int, paddle_y: int) -> int:
        """Deterministic reflection angle based on hit offset along paddle."""
        ball_center = ball_y + 1.0
        paddle_center = paddle_y + (PADDLE_H - 1) / 2.0  # paddle_y + 6.5
        rel = ball_center - paddle_center
        if rel < -4.5:
            return -2
        if rel < -1.5:
            return -1
        if rel <= 1.5:
            return 0
        if rel <= 4.5:
            return 1
        return 2

    def _update_left_ai(self) -> None:
        """Left AI paddle tracks the ball with optional stochastic jitter."""
        s = self.state
        if self.ai_jitter > 0.0 and self.rng.random() < self.ai_jitter:
            # Occasional hesitation or 1-step jitter to diversify rally outcomes
            delta = int(self.rng.choice([-AI_SPEED, 0, AI_SPEED]))
            s.left_y = int(np.clip(s.left_y + delta, PADDLE_MIN_Y, PADDLE_MAX_Y))
            return

        # Deterministic AI tracking: tracks ball center smoothly when ball moves left,
        # and gently drifts toward court center when ball moves right.
        paddle_center = s.left_y + PADDLE_H // 2
        if s.ball_vx < 0:
            target_y = s.ball_y + 1
            deadzone = 2
        else:
            target_y = FRAME_H // 2
            deadzone = 3

        diff = target_y - paddle_center
        if diff < -deadzone:
            s.left_y = max(PADDLE_MIN_Y, s.left_y - AI_SPEED)
        elif diff > deadzone:
            s.left_y = min(PADDLE_MAX_Y, s.left_y + AI_SPEED)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance Pong physics by 1 frame given right-paddle action in {0, 1, 2}."""
        action = int(action)
        if action not in (ACTION_NONE, ACTION_UP, ACTION_DOWN):
            raise ValueError(f"Invalid action {action}; expected 0, 1, or 2.")

        s = self.state
        prev_ball_x = s.ball_x

        # 1. Update right (player) paddle strictly according to action
        if action == ACTION_UP:
            s.right_y = max(PADDLE_MIN_Y, s.right_y - PLAYER_SPEED)
        elif action == ACTION_DOWN:
            s.right_y = min(PADDLE_MAX_Y, s.right_y + PLAYER_SPEED)

        # 2. Update left (AI) paddle
        self._update_left_ai()

        # 3. Advance ball position
        s.ball_x += s.ball_vx
        s.ball_y += s.ball_vy

        event = "move"

        # 4. Top / bottom wall reflection
        if s.ball_y < BALL_MIN_Y:
            s.ball_y = BALL_MIN_Y + (BALL_MIN_Y - s.ball_y)
            s.ball_vy = abs(s.ball_vy) if s.ball_vy != 0 else 1
            event = "wall_bounce"
        elif s.ball_y > BALL_MAX_Y:
            s.ball_y = BALL_MAX_Y - (s.ball_y - BALL_MAX_Y)
            s.ball_vy = -abs(s.ball_vy) if s.ball_vy != 0 else -1
            event = "wall_bounce"

        # 5. Right paddle collision (ball moving right)
        right_hit_x = (
            s.ball_vx > 0
            and (s.ball_x + BALL_SIZE >= RIGHT_PADDLE_X)
            and (prev_ball_x + BALL_SIZE <= RIGHT_PADDLE_X)
        )
        right_hit_y = (s.ball_y + BALL_SIZE > s.right_y) and (
            s.ball_y < s.right_y + PADDLE_H
        )
        if right_hit_x and right_hit_y:
            s.ball_x = RIGHT_PADDLE_X - BALL_SIZE  # 73
            s.ball_vx = -BALL_SPEED_X
            s.ball_vy = self._bounce_vy(s.ball_y, s.right_y)
            s.rallies += 1
            event = "right_paddle_bounce"

        # 6. Left paddle collision (ball moving left)
        left_face = LEFT_PADDLE_X + PADDLE_W  # 8
        left_hit_x = (
            s.ball_vx < 0
            and (s.ball_x <= left_face)
            and (prev_ball_x >= left_face)
        )
        left_hit_y = (s.ball_y + BALL_SIZE > s.left_y) and (
            s.ball_y < s.left_y + PADDLE_H
        )
        if left_hit_x and left_hit_y:
            s.ball_x = left_face  # 8
            s.ball_vx = BALL_SPEED_X
            s.ball_vy = self._bounce_vy(s.ball_y, s.left_y)
            s.rallies += 1
            event = "left_paddle_bounce"

        # 7. Scoring & deterministic center serve
        reward = 0.0
        if s.ball_x < 0:
            s.right_score += 1
            reward = 1.0
            event = "right_score"
            self._serve_ball(toward_right=True)
        elif s.ball_x + BALL_SIZE > FRAME_W:
            s.left_score += 1
            reward = -1.0
            event = "left_score"
            self._serve_ball(toward_right=False)

        s.step_count += 1
        done = (
            s.left_score >= self.max_points
            or s.right_score >= self.max_points
            or s.step_count >= self.max_steps
        )
        info = {
            "event": event,
            "left_score": s.left_score,
            "right_score": s.right_score,
            "rallies": s.rallies,
            "step_count": s.step_count,
            "ball_xy": (s.ball_x, s.ball_y),
            "right_y": s.right_y,
            "left_y": s.left_y,
        }
        return self.render(), reward, done, info

    def _serve_ball(self, toward_right: bool) -> None:
        """Deterministic serve from center based only on visible paddle position."""
        s = self.state
        s.ball_x = 40
        s.ball_y = 40
        s.ball_vx = BALL_SPEED_X if toward_right else -BALL_SPEED_X
        # Deterministic vertical direction from right paddle position (observable in pixels!)
        s.ball_vy = -1 if s.right_y < 35 else 1

    def render(self, out: np.ndarray | None = None) -> np.ndarray:
        """Render current state to an 84x84 binary uint8 frame (0 or 255)."""
        if out is None:
            frame = STATIC_BACKGROUND.copy()
        else:
            np.copyto(out, STATIC_BACKGROUND)
            frame = out

        s = self.state
        # Left paddle
        frame[s.left_y : s.left_y + PADDLE_H, LEFT_PADDLE_X : LEFT_PADDLE_X + PADDLE_W] = 255
        # Right paddle
        frame[
            s.right_y : s.right_y + PADDLE_H,
            RIGHT_PADDLE_X : RIGHT_PADDLE_X + PADDLE_W,
        ] = 255
        # Ball (clamped safely for edge frames)
        y0 = max(0, s.ball_y)
        y1 = min(FRAME_H, s.ball_y + BALL_SIZE)
        x0 = max(0, s.ball_x)
        x1 = min(FRAME_W, s.ball_x + BALL_SIZE)
        if y0 < y1 and x0 < x1:
            frame[y0:y1, x0:x1] = 255
        return frame


def choose_player_action(
    env: PongEnv,
    policy: str,
    rng: np.random.Generator,
    sticky_state: dict[str, int],
) -> int:
    """Diverse player policies for rich dataset collection.

    Policies:
    - 'tracker': semi-optimal tracker with random hit offset -> produces long rallies
      and hits across all paddle regions (diverse bounce angles).
    - 'sticky_random': holds random UP/DOWN/NONE for 3..14 frames -> decouples action
      signal from ball location so the neural model MUST obey a_t.
    - 'mixed': alternates smoothly between tracking and exploration/misses -> covers
      wall bounces, paddle bounces, AND scoring/center serves.
    """
    s = env.state
    if policy == "sticky_random":
        if sticky_state["rem"] <= 0:
            sticky_state["act"] = int(rng.choice([ACTION_NONE, ACTION_UP, ACTION_DOWN], p=[0.25, 0.375, 0.375]))
            sticky_state["rem"] = int(rng.integers(3, 15))
        sticky_state["rem"] -= 1
        return sticky_state["act"]

    if policy == "mixed":
        # Re-roll mode every 20..50 frames
        if sticky_state.get("mode_rem", 0) <= 0:
            sticky_state["mode"] = int(rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15]))
            sticky_state["mode_rem"] = int(rng.integers(20, 50))
            sticky_state["offset"] = int(rng.integers(-6, 7))
        sticky_state["mode_rem"] -= 1
        if sticky_state["mode"] == 1:
            return choose_player_action(env, "sticky_random", rng, sticky_state)
        if sticky_state["mode"] == 2:
            # Deliberately move away / hold still occasionally to produce scoring & serves
            return ACTION_NONE if rng.random() < 0.7 else int(rng.integers(0, 3))

    # Default 'tracker' (or mode 0 of 'mixed')
    if sticky_state.get("offset_rem", 0) <= 0:
        sticky_state["offset"] = int(rng.integers(-6, 7))
        sticky_state["offset_rem"] = int(rng.integers(15, 45))
    sticky_state["offset_rem"] -= 1

    # Small random exploration within tracker
    if rng.random() < 0.08:
        return int(rng.integers(0, 3))

    paddle_center = s.right_y + PADDLE_H // 2
    target = s.ball_y + 1 + sticky_state["offset"]
    diff = target - paddle_center
    if diff < -2:
        return ACTION_UP
    if diff > 2:
        return ACTION_DOWN
    return ACTION_NONE
