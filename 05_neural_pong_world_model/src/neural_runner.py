"""Pure Neural Pong World Model Runtime — ZERO Game Engine Code.

CRITICAL INVARIANT:
This module MUST NEVER import `pong_env` or contain any Pong physics logic.
All frame generation runs 100% autoregressively through the trained neural network weights
(`artifacts/neural_pong_best.pt`), seeded solely by 4 initial frames (`artifacts/seed_frames.npy`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.model import MODEL_CONFIGS, ModelSize, NeuralPongWorldModel, configure_cuda_backend


def load_world_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[NeuralPongWorldModel, dict[str, Any]]:
    """Load a trained NeuralPongWorldModel checkpoint safely on CUDA or CPU."""
    configure_cuda_backend()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_size: ModelSize = ckpt.get("model_size", "small")
    if model_size not in MODEL_CONFIGS:
        model_size = "small"

    model = NeuralPongWorldModel(model_size).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "model_size": model_size,
        "param_count": model.count_parameters(),
        "device": str(device),
        "epoch": ckpt.get("epoch", None),
        "val_metrics": ckpt.get("val_metrics", {}),
        "checkpoint_path": str(ckpt_path),
    }
    return model, meta


class NeuralPongSession:
    """Interactive autoregressive Pong session running 100% inside a Neural Network."""

    def __init__(
        self,
        model: NeuralPongWorldModel,
        seed_frames: np.ndarray,
        device: torch.device | str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.use_fp16 = self.device.type == "cuda"
        self.threshold = float(threshold)

        # seed_frames: (num_seeds, 4, 84, 84) uint8 or float
        arr = np.asarray(seed_frames)
        if arr.ndim == 3:
            arr = arr[None, ...]  # (1, 4, 84, 84)
        if arr.shape[1:] != (4, 84, 84):
            raise ValueError(f"Expected seed_frames shape (N, 4, 84, 84), got {arr.shape}")
        self.seed_frames = arr
        self.num_seeds = int(arr.shape[0])

        self.seed_idx = 0
        self.step_count = 0
        self.buffer: torch.Tensor | None = None
        self.reset(0)

    def reset(self, seed_idx: int = 0) -> np.ndarray:
        """Load 4 initial frames from seed bank; after this, 100% neural autoregressive."""
        self.seed_idx = int(seed_idx) % self.num_seeds
        raw4 = self.seed_frames[self.seed_idx]
        if raw4.dtype == np.uint8:
            f4 = raw4.astype(np.float32) * (1.0 / 255.0)
        else:
            f4 = np.clip(raw4.astype(np.float32), 0.0, 1.0)
        self.buffer = torch.from_numpy(f4).unsqueeze(0).to(self.device)  # (1, 4, 84, 84)
        self.step_count = 0
        return (f4[-1] >= self.threshold).astype(np.uint8) * 255

    @torch.no_grad()
    def step(self, action: int) -> tuple[np.ndarray, float]:
        """Run one pure neural autoregressive step given action in {0, 1, 2}."""
        if self.buffer is None:
            self.reset(0)
        assert self.buffer is not None

        act_int = int(action)
        if act_int not in (0, 1, 2):
            act_int = 0
        act_t = torch.tensor([act_int], dtype=torch.long, device=self.device)

        t0 = time.perf_counter()
        if self.use_fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                next_bin, _probs = self.model.predict_next_frame(
                    self.buffer, act_t, threshold=self.threshold
                )
        else:
            next_bin, _probs = self.model.predict_next_frame(
                self.buffer, act_t, threshold=self.threshold
            )
        if self.use_fp16:
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # Autoregressive feedback: drop oldest frame, append neural prediction
        next_bin_f32 = next_bin.float()
        self.buffer = torch.cat([self.buffer[:, 1:, :, :], next_bin_f32], dim=1)
        self.step_count += 1

        frame_u8 = (next_bin_f32[0, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
        return frame_u8, infer_ms


@torch.no_grad()
def autoregressive_rollout(
    model: NeuralPongWorldModel,
    initial_frames: torch.Tensor,
    actions: torch.Tensor,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched pure neural multi-step autoregressive rollout with ZERO game engine.

    Args:
        model: Trained NeuralPongWorldModel in eval mode.
        initial_frames: (B, 4, 84, 84) float tensor in [0, 1].
        actions: (B, T) int64 tensor of actions in {0, 1, 2}.
        threshold: Binarization threshold for feeding predicted frame back into buffer.
        use_amp: Use FP16 autocast when running on CUDA.

    Returns:
        pred_frames: (B, T, 84, 84) binary float32 tensor in {0.0, 1.0}.
        pred_probs: (B, T, 84, 84) probability float32 tensor in [0.0, 1.0].
    """
    model.eval()
    device = next(model.parameters()).device
    buf = initial_frames.to(device=device, dtype=torch.float32)
    acts = actions.to(device=device, dtype=torch.long)
    b, horizon = acts.shape
    amp_enabled = use_amp and (device.type == "cuda")

    frames_out = torch.empty((b, horizon, 84, 84), dtype=torch.float32, device="cpu")
    probs_out = torch.empty((b, horizon, 84, 84), dtype=torch.float32, device="cpu")

    for t in range(horizon):
        if amp_enabled:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                next_bin, next_prob = model.predict_next_frame(
                    buf, acts[:, t], threshold=threshold
                )
        else:
            next_bin, next_prob = model.predict_next_frame(
                buf, acts[:, t], threshold=threshold
            )
        next_bin_f32 = next_bin.float()
        frames_out[:, t] = next_bin_f32[:, 0].cpu()
        probs_out[:, t] = next_prob.float()[:, 0].cpu()
        buf = torch.cat([buf[:, 1:, :, :], next_bin_f32], dim=1)

    return frames_out, probs_out
