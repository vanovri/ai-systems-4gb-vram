"""Memory-mapped dataset for 4-frame Neural Pong World Model with strict episode boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_valid_indices(
    episode_starts: np.ndarray,
    stack_size: int = 4,
    rollout_steps: int = 1,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> np.ndarray:
    """Compute anchor indices `t` whose window [t-(stack_size-1) .. t+rollout_steps]
    lies strictly within a single episode and within [start_idx, end_idx).

    Convention:
    - frames[t] is x_t
    - actions[t] transforms x_t -> x_{t+1}
    - episode_starts[i] == True means frames[i] starts a fresh episode
    - Valid window [t - (stack_size - 1) .. t + rollout_steps] allows
      episode_starts[t - (stack_size - 1)] to be True, while every subsequent
      index in [t - (stack_size - 2) .. t + rollout_steps] MUST be False.
    """
    n_total = len(episode_starts)
    if end_idx is None:
        end_idx = n_total
    end_idx = min(end_idx, n_total)

    min_t = max(start_idx + stack_size - 1, stack_size - 1)
    max_t = end_idx - rollout_steps - 1  # inclusive
    if max_t < min_t:
        return np.empty(0, dtype=np.int64)

    ep_int = np.asarray(episode_starts, dtype=np.int32)
    # Prefix sum over episode_starts: csum[b+1] - csum[a] == count of True in [a..b]
    csum = np.empty(n_total + 1, dtype=np.int64)
    csum[0] = 0
    np.cumsum(ep_int, out=csum[1:])

    candidates = np.arange(min_t, max_t + 1, dtype=np.int64)
    win_start = candidates - (stack_size - 2)  # exclusive of oldest frame in stack
    win_end = candidates + rollout_steps + 1  # exclusive end for prefix sum
    crossings = csum[win_end] - csum[win_start]
    return candidates[crossings == 0]


class PongSequenceDataset(Dataset):
    """Memory-mapped PyTorch dataset for single-step or K-step Neural Pong training.

    Uses `np.load(..., mmap_mode='r')` so system RAM stays well below 500 MB even for
    200,000+ 84x84 uint8 frames.
    """

    def __init__(
        self,
        data_dir: str | Path,
        start_idx: int = 0,
        end_idx: int | None = None,
        stack_size: int = 4,
        rollout_steps: int = 1,
        augment_noise: bool = False,
        bitflip_prob_max: float = 0.004,
        drop_pixel_prob: float = 0.03,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.frames = np.load(self.data_dir / "frames.npy", mmap_mode="r")
        self.actions = np.load(self.data_dir / "actions.npy", mmap_mode="r")
        self.episode_starts = np.load(self.data_dir / "episode_starts.npy")

        self.stack_size = int(stack_size)
        self.rollout_steps = int(rollout_steps)
        self.augment_noise = bool(augment_noise)
        self.bitflip_prob_max = float(bitflip_prob_max)
        self.drop_pixel_prob = float(drop_pixel_prob)
        self.seed = int(seed)

        self.indices = compute_valid_indices(
            episode_starts=self.episode_starts,
            stack_size=self.stack_size,
            rollout_steps=self.rollout_steps,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def _corrupt_inputs(self, x: np.ndarray) -> np.ndarray:
        """Light GameNGen-style input noise augmentation for autoregressive robustness.

        Applied with 50% probability so the model sees both clean binary frames
        and slightly corrupted frames (bit-flips / foreground pixel dropouts /
        small soft Gaussian-like jitter), teaching it to denoise rollout drift.
        """
        if np.random.random() >= 0.5:
            return x

        x = x.copy()
        mode = np.random.random()
        if mode < 0.65:
            # Random sparse binary bit-flips (0.05% .. bitflip_prob_max)
            p = np.random.uniform(0.0005, self.bitflip_prob_max)
            flip_mask = np.random.random(x.shape) < p
            x[flip_mask] = 1.0 - x[flip_mask]
        elif mode < 0.85:
            # Slight foreground dropout (simulates faint pixel loss in rollout)
            drop_mask = (x > 0.5) & (np.random.random(x.shape) < self.drop_pixel_prob)
            x[drop_mask] = 0.0
        else:
            # Tiny additive sensor noise clipped to [0, 1]
            sigma = np.random.uniform(0.02, 0.08)
            x = np.clip(x + np.random.normal(0.0, sigma, size=x.shape).astype(np.float32), 0.0, 1.0)
        return x

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        # Stack of 4 consecutive frames: [t-3, t-2, t-1, t]
        s0 = t - (self.stack_size - 1)
        input_frames = np.array(self.frames[s0 : t + 1], dtype=np.float32) * (1.0 / 255.0)
        if self.augment_noise:
            input_frames = self._corrupt_inputs(input_frames)

        if self.rollout_steps == 1:
            action = np.int64(self.actions[t])
            target = (
                np.array(self.frames[t + 1 : t + 2], dtype=np.float32) * (1.0 / 255.0)
            )
            return (
                torch.from_numpy(input_frames),
                torch.tensor(action, dtype=torch.long),
                torch.from_numpy(target),
            )

        # Multi-step unroll targets: actions [t .. t+K-1], targets [t+1 .. t+K]
        actions = np.array(self.actions[t : t + self.rollout_steps], dtype=np.int64)
        targets = (
            np.array(self.frames[t + 1 : t + 1 + self.rollout_steps], dtype=np.float32)
            * (1.0 / 255.0)
        )
        return (
            torch.from_numpy(input_frames),
            torch.from_numpy(actions),
            torch.from_numpy(targets),
        )


def make_split_bounds(
    episode_starts: np.ndarray,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> dict[Literal["train", "val", "test"], tuple[int, int]]:
    """Split frame range strictly on episode boundaries so train/val/test never share an episode."""
    n = len(episode_starts)
    ep_positions = np.flatnonzero(episode_starts)
    if len(ep_positions) < 3:
        # Fallback for tiny synthetic unit-test datasets
        t_end = max(1, int(n * (1.0 - val_frac - test_frac)))
        v_end = max(t_end, int(n * (1.0 - test_frac)))
        return {"train": (0, t_end), "val": (t_end, v_end), "test": (v_end, n)}

    target_train_end = int(n * (1.0 - val_frac - test_frac))
    target_val_end = int(n * (1.0 - test_frac))

    train_ep_idx = int(np.searchsorted(ep_positions, target_train_end, side="left"))
    train_ep_idx = min(max(train_ep_idx, 1), len(ep_positions) - 2)
    train_end = int(ep_positions[train_ep_idx])

    val_ep_idx = int(np.searchsorted(ep_positions, target_val_end, side="left"))
    val_ep_idx = min(max(val_ep_idx, train_ep_idx + 1), len(ep_positions) - 1)
    val_end = int(ep_positions[val_ep_idx])

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n),
    }
