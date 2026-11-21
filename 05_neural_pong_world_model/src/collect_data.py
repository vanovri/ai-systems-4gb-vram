"""Collect memory-mapped 84x84 Pong dataset (frames.npy, actions.npy, episode_starts.npy).

Writes directly to disk via `np.lib.format.open_memmap` so peak RAM stays well under 500 MB.
Also exports clean held-out 4-frame play seeds (`artifacts/seed_frames.npy` +
`artifacts/seed_states.json`) for interactive Neural-Only and Split-Screen play.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from src.dataset import compute_valid_indices, make_split_bounds
from src.pong_env import (
    ACTION_NAMES,
    FRAME_H,
    FRAME_W,
    PongEnv,
    choose_player_action,
)


def collect_dataset(
    out_dir: str | Path,
    num_frames: int = 140_000,
    seed: int = 2026,
    ai_jitter: float = 0.04,
    max_points: int = 5,
    max_steps: int = 450,
    seed_pack_dir: str | Path | None = None,
    num_play_seeds: int = 16,
) -> dict[str, Any]:
    """Generate `num_frames` of diverse Pong trajectories directly to memmap `.npy` files."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    frames_mm = open_memmap(
        out_path / "frames.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(num_frames, FRAME_H, FRAME_W),
    )
    actions_mm = open_memmap(
        out_path / "actions.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(num_frames,),
    )
    ep_starts_mm = open_memmap(
        out_path / "episode_starts.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(num_frames,),
    )

    rng = np.random.default_rng(seed)
    # Use small AI jitter in train episodes so left-side misses & varied rallies occur,
    # while keeping left AI mostly deterministic (~96% exact tracking) so the model learns
    # a crisp left paddle!
    env = PongEnv(seed=seed, ai_jitter=ai_jitter, max_points=max_points, max_steps=max_steps)

    policies = ["tracker", "sticky_random", "mixed"]
    policy_probs = [0.40, 0.35, 0.25]
    policy_counts: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    event_counts: Counter[str] = Counter()

    idx = 0
    episode_id = 0

    while idx < num_frames:
        # In the final 10% (val/test), use ai_jitter=0.0 so held-out deterministic
        # evaluation comparisons against ground truth have zero stochastic mismatch.
        env.ai_jitter = 0.0 if idx >= int(num_frames * 0.88) else ai_jitter

        policy = str(rng.choice(policies, p=policy_probs))
        policy_counts[policy] += 1
        sticky_state = {"act": 0, "rem": 0, "offset": 0, "offset_rem": 0, "mode": 0, "mode_rem": 0}

        frame = env.reset(randomize_positions=True)
        episode_id += 1
        is_start = True

        while idx < num_frames:
            frames_mm[idx] = frame
            ep_starts_mm[idx] = is_start
            is_start = False

            action = choose_player_action(env, policy=policy, rng=rng, sticky_state=sticky_state)
            actions_mm[idx] = action
            action_counts[action] += 1

            next_frame, _reward, done, info = env.step(action)
            event_counts[info["event"]] += 1
            idx += 1

            if done:
                break
            frame = next_frame

    frames_mm.flush()
    actions_mm.flush()
    ep_starts_mm.flush()
    del frames_mm, actions_mm, ep_starts_mm

    # Compute exact episode-aligned train/val/test splits
    ep_starts = np.load(out_path / "episode_starts.npy")
    splits = make_split_bounds(ep_starts, val_frac=0.05, test_frac=0.05)

    split_valid_1step = {
        name: int(len(compute_valid_indices(ep_starts, stack_size=4, rollout_steps=1, start_idx=a, end_idx=b)))
        for name, (a, b) in splits.items()
    }
    split_valid_4step = {
        name: int(len(compute_valid_indices(ep_starts, stack_size=4, rollout_steps=4, start_idx=a, end_idx=b)))
        for name, (a, b) in splits.items()
    }

    elapsed = time.perf_counter() - t0
    meta = {
        "num_frames": num_frames,
        "frame_shape": [FRAME_H, FRAME_W],
        "dtype": "uint8",
        "num_episodes": episode_id,
        "seed": seed,
        "ai_jitter_train": ai_jitter,
        "elapsed_sec": round(elapsed, 2),
        "frames_per_sec": round(num_frames / max(elapsed, 1e-6), 1),
        "policy_episodes": dict(policy_counts),
        "action_distribution": {
            ACTION_NAMES[k]: int(action_counts[k]) for k in sorted(ACTION_NAMES)
        },
        "events": dict(event_counts),
        "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
        "valid_1step_samples": split_valid_1step,
        "valid_4step_samples": split_valid_4step,
    }

    with open(out_path / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if seed_pack_dir is not None:
        generate_play_seeds(seed_pack_dir, num_seeds=num_play_seeds, seed=seed + 99_991)

    return meta


def generate_play_seeds(
    artifacts_dir: str | Path,
    num_seeds: int = 16,
    seed: int = 99_991,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate clean 4-frame initial seeds + matching env state (for optional split-screen sync).

    In Neural-Only play mode, ONLY `seed_frames.npy` is loaded (zero game engine imported!).
    In Split-Screen comparison mode, `seed_states.json` lets the real engine start from
    the exact same 4th-frame Markov state.
    """
    art_path = Path(artifacts_dir)
    art_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    seed_frames = np.zeros((num_seeds, 4, FRAME_H, FRAME_W), dtype=np.uint8)
    seed_states: list[dict[str, Any]] = []

    for i in range(num_seeds):
        env = PongEnv(seed=seed + i * 37, ai_jitter=0.0, max_points=99, max_steps=10_000)
        # Put ball cleanly in mid-court moving toward either paddle for immediate fun play
        left_y = int(rng.integers(14, 54))
        right_y = int(rng.integers(14, 54))
        ball_x = int(rng.integers(28, 52))
        ball_y = int(rng.integers(18, 62))
        ball_vx = 2 if (i % 2 == 0) else -2
        ball_vy = int(rng.choice([-2, -1, 1, 2]))

        from src.pong_env import PongState

        init_state = PongState(
            left_y=left_y,
            right_y=right_y,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_vx=ball_vx,
            ball_vy=ball_vy,
        )
        f0 = env.reset(state=init_state)
        seed_frames[i, 0] = f0
        for step_k in range(1, 4):
            fk, _, _, _ = env.step(0)
            seed_frames[i, step_k] = fk
        seed_states.append(env.get_state().to_dict())

    np.save(art_path / "seed_frames.npy", seed_frames)
    with open(art_path / "seed_states.json", "w", encoding="utf-8") as f:
        json.dump(seed_states, f, indent=2)

    return seed_frames, seed_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect deterministic 84x84 Pong memmap dataset.")
    parser.add_argument("--out-dir", type=str, default="data/pong_140k")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--num-frames", type=int, default=140_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ai-jitter", type=float, default=0.04)
    parser.add_argument("--num-play-seeds", type=int, default=16)
    args = parser.parse_args()

    meta = collect_dataset(
        out_dir=args.out_dir,
        num_frames=args.num_frames,
        seed=args.seed,
        ai_jitter=args.ai_jitter,
        seed_pack_dir=args.artifacts_dir,
        num_play_seeds=args.num_play_seeds,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
