"""Comprehensive Teacher-Forced & Multi-Step Autoregressive Evaluation for Neural Pong World Model.

Computes real metrics on held-out test data:
1. Teacher-forced 1-step:
   - Unweighted BCE loss & pseudo-perplexity exp(BCE)
   - Full-frame & Dynamic-region (paddles + ball only) Pixel F1 / IoU / Precision / Recall
   - Ball Centroid Error (px) & Ball Detection Rate (%)
   - Counterfactual Paddle Responsiveness Accuracy (%) with ZERO game engine
     (identical 4-frame context tested under UP, DOWN, NONE actions)
2. Multi-step Autoregressive Rollout (10, 30, 60 frames with ZERO game engine!):
   - Full-frame & Dynamic IoU decay curve over 60 steps
   - Ball & Right-Paddle Centroid Error decay over 60 steps
   - Real single-frame interactive FPS & latency (ms) on CUDA/CPU
3. Visual Artifacts:
   - `reports/eval_rollout.json`
   - `reports/rollout_comparison.gif` (side-by-side GT vs Pure Neural vs Diff)
   - `reports/rollout_grid.png` (keyframe comparison grid + 60-step decay plots)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import PongSequenceDataset
from src.neural_runner import autoregressive_rollout, load_world_model
from src.pong_env import (
    ACTION_DOWN,
    ACTION_NAMES,
    ACTION_NONE,
    ACTION_UP,
    FRAME_H,
    FRAME_W,
    LEFT_PADDLE_X,
    PADDLE_H,
    PADDLE_W,
    PLAY_BOTTOM,
    PLAY_TOP,
    RIGHT_PADDLE_X,
    STATIC_BACKGROUND,
    PongEnv,
    PongState,
    choose_player_action,
)

STATIC_MASK = STATIC_BACKGROUND > 127
DYNAMIC_MASK = ~STATIC_MASK


def extract_paddle_cy(frame_bin: np.ndarray, side: str = "right") -> float | None:
    """Extract vertical centroid (y in [2, 81]) of left or right paddle from binary 84x84 frame."""
    x0 = RIGHT_PADDLE_X if side == "right" else LEFT_PADDLE_X
    patch = frame_bin[PLAY_TOP:PLAY_BOTTOM, x0 : x0 + PADDLE_W] > 0.5
    ys = np.where(patch)[0]
    if ys.size < 6:
        return None
    return float(ys.mean() + PLAY_TOP)


def _box3x3_sum(grid: np.ndarray) -> np.ndarray:
    """Fast pure-NumPy 3x3 valid box-sum over a 2D float array."""
    p = np.pad(grid.astype(np.float32), ((1, 1), (1, 1)), mode="constant")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )


def extract_ball_centroid(frame_bin: np.ndarray) -> tuple[float, float] | None:
    """Extract (cx, cy) centroid of the 3x3 ball in the court playfield (cols 8..75).

    Immune to the 2-px dashed center line because a 3x3 ball has 3x3 sum up to 9
    plus >=3 non-static pixels sticking out of the 2-px center line, whereas pure
    dashes have 3x3 sum <= 6 and 0 non-static pixels.
    """
    x_min, x_max = LEFT_PADDLE_X + PADDLE_W, RIGHT_PADDLE_X  # 8..76
    y_min, y_max = PLAY_TOP, PLAY_BOTTOM  # 2..82

    field = (frame_bin[y_min:y_max, x_min:x_max] > 0.5).astype(np.float32)
    non_static = field * (~STATIC_MASK[y_min:y_max, x_min:x_max]).astype(np.float32)

    if non_static.sum() < 2.0:
        return None

    # Score combines full 3x3 white density + non-static 3x3 density (rejects pure dashes)
    score = _box3x3_sum(field) + 1.25 * _box3x3_sum(non_static)
    # Require at least some non-static evidence inside the 3x3 window
    ns_sum = _box3x3_sum(non_static)
    score = np.where(ns_sum >= 2.0, score, 0.0)

    peak_val = float(score.max())
    if peak_val < 5.5:
        return None

    py, px = np.unravel_index(int(np.argmax(score)), score.shape)
    # Compute centroid in the 5x5 local window around the peak
    ry0, ry1 = max(0, py - 2), min(field.shape[0], py + 3)
    rx0, rx1 = max(0, px - 2), min(field.shape[1], px + 3)
    local_w = field[ry0:ry1, rx0:rx1].copy()
    # Mask static center-line pixels that lie outside the inner 3x3 ball footprint
    iy0, iy1 = max(0, py - 1) - ry0, min(field.shape[0], py + 2) - ry0
    ix0, ix1 = max(0, px - 1) - rx0, min(field.shape[1], px + 2) - rx0
    inner_mask = np.zeros_like(local_w, dtype=bool)
    inner_mask[iy0:iy1, ix0:ix1] = True
    local_static = STATIC_MASK[y_min + ry0 : y_min + ry1, x_min + rx0 : x_min + rx1]
    local_w = np.where(inner_mask | (~local_static), local_w, 0.0)

    total = float(local_w.sum())
    if total < 2.0:
        return None
    yy, xx = np.indices(local_w.shape)
    cy = float((yy * local_w).sum() / total + y_min + ry0)
    cx = float((xx * local_w).sum() / total + x_min + rx0)
    return (cx, cy)


def binary_overlap_metrics(
    pred_bin: np.ndarray, gt_bin: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Compute IoU, F1, Precision, Recall on binary masks (optionally restricted to `mask`)."""
    p = pred_bin > 0.5
    g = gt_bin > 0.5
    if mask is not None:
        p = p & mask
        g = g & mask
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    eps = 1e-8
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


@torch.no_grad()
def evaluate_teacher_forced(
    model: torch.nn.Module,
    data_dir: Path,
    test_bounds: tuple[int, int],
    max_samples: int = 2000,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate 1-step teacher-forced accuracy + counterfactual paddle responsiveness."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    use_amp = device.type == "cuda"

    ds = PongSequenceDataset(
        data_dir=data_dir,
        start_idx=test_bounds[0],
        end_idx=test_bounds[1],
        stack_size=4,
        rollout_steps=1,
        augment_noise=False,
    )
    n_eval = min(len(ds), max_samples)
    # Evenly stride across the full held-out test split
    sample_ids = np.linspace(0, len(ds) - 1, n_eval, dtype=np.int64)

    bce_sum = 0.0
    total_px = 0
    full_ious, full_f1s, full_precs, full_recs = [], [], [], []
    dyn_ious, dyn_f1s = [], []
    ball_errors: list[float] = []
    ball_gt_count = 0
    ball_det_count = 0

    for start in range(0, n_eval, batch_size):
        chunk_ids = sample_ids[start : start + batch_size]
        xs, acts, ys = zip(*(ds[int(i)] for i in chunk_ids))
        x_t = torch.stack(xs).to(device)
        a_t = torch.stack(acts).to(device)
        y_t = torch.stack(ys).to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x_t, a_t)
        logits_f32 = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits_f32, y_t.float(), reduction="sum")
        bce_sum += float(bce.item())
        total_px += int(y_t.numel())

        preds_np = (torch.sigmoid(logits_f32)[:, 0].cpu().numpy() >= 0.5).astype(np.float32)
        gt_np = (y_t[:, 0].cpu().numpy() >= 0.5).astype(np.float32)

        for p_f, g_f in zip(preds_np, gt_np):
            fm = binary_overlap_metrics(p_f, g_f)
            dm = binary_overlap_metrics(p_f, g_f, mask=DYNAMIC_MASK)
            full_ious.append(fm["iou"])
            full_f1s.append(fm["f1"])
            full_precs.append(fm["precision"])
            full_recs.append(fm["recall"])
            dyn_ious.append(dm["iou"])
            dyn_f1s.append(dm["f1"])

            gt_ball = extract_ball_centroid(g_f)
            if gt_ball is not None:
                ball_gt_count += 1
                pred_ball = extract_ball_centroid(p_f)
                if pred_ball is not None:
                    ball_det_count += 1
                    err = math.hypot(pred_ball[0] - gt_ball[0], pred_ball[1] - gt_ball[1])
                    ball_errors.append(err)

    raw_bce = bce_sum / max(total_px, 1)
    pseudo_ppl = math.exp(min(raw_bce, 20.0))

    # Counterfactual Paddle Responsiveness Test (ZERO game engine!)
    # Pick up to 300 held-out states where right paddle is comfortably away from walls.
    cf_frames: list[torch.Tensor] = []
    cf_prev_cy: list[float] = []
    for idx in sample_ids:
        x_i, _, _ = ds[int(idx)]
        prev_cy = extract_paddle_cy(x_i[-1].numpy(), side="right")
        # Paddle center y in [14, 70] -> top y in [7, 63], safely away from top/bottom clamps
        if prev_cy is not None and 14.0 <= prev_cy <= 70.0:
            cf_frames.append(x_i)
            cf_prev_cy.append(prev_cy)
            if len(cf_frames) >= 300:
                break

    up_ok = down_ok = none_ok = strict_order_ok = 0
    up_deltas, down_deltas, none_deltas = [], [], []
    n_cf = len(cf_frames)
    if n_cf > 0:
        x_cf = torch.stack(cf_frames).to(device)
        preds_by_act: dict[int, np.ndarray] = {}
        for act_val in (ACTION_NONE, ACTION_UP, ACTION_DOWN):
            a_vec = torch.full((n_cf,), act_val, dtype=torch.long, device=device)
            chunks = []
            for s in range(0, n_cf, batch_size):
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    lg = model(x_cf[s : s + batch_size], a_vec[s : s + batch_size])
                chunks.append((torch.sigmoid(lg.float())[:, 0].cpu().numpy() >= 0.5).astype(np.float32))
            preds_by_act[act_val] = np.concatenate(chunks, axis=0)

        for i in range(n_cf):
            cy0 = cf_prev_cy[i]
            cy_none = extract_paddle_cy(preds_by_act[ACTION_NONE][i], side="right")
            cy_up = extract_paddle_cy(preds_by_act[ACTION_UP][i], side="right")
            cy_down = extract_paddle_cy(preds_by_act[ACTION_DOWN][i], side="right")
            if cy_up is not None:
                d_up = cy_up - cy0
                up_deltas.append(d_up)
                if d_up < -0.5:
                    up_ok += 1
            if cy_down is not None:
                d_dn = cy_down - cy0
                down_deltas.append(d_dn)
                if d_dn > 0.5:
                    down_ok += 1
            if cy_none is not None:
                d_no = cy_none - cy0
                none_deltas.append(d_no)
                if abs(d_no) <= 0.5:
                    none_ok += 1
            if (
                cy_up is not None
                and cy_none is not None
                and cy_down is not None
                and (cy_up < cy_none - 0.5)
                and (cy_down > cy_none + 0.5)
            ):
                strict_order_ok += 1

    ball_err_arr = np.array(ball_errors, dtype=np.float64) if ball_errors else np.array([float("nan")])
    return {
        "num_eval_samples": int(n_eval),
        "bce_loss": round(raw_bce, 6),
        "pseudo_perplexity": round(pseudo_ppl, 6),
        "full_frame_iou": round(float(np.mean(full_ious)), 5),
        "full_frame_f1": round(float(np.mean(full_f1s)), 5),
        "full_frame_precision": round(float(np.mean(full_precs)), 5),
        "full_frame_recall": round(float(np.mean(full_recs)), 5),
        "dynamic_region_iou": round(float(np.mean(dyn_ious)), 5),
        "dynamic_region_f1": round(float(np.mean(dyn_f1s)), 5),
        "ball_centroid_error_px_mean": round(float(np.nanmean(ball_err_arr)), 4),
        "ball_centroid_error_px_median": round(float(np.nanmedian(ball_err_arr)), 4),
        "ball_within_1px_pct": round(float(np.mean(ball_err_arr <= 1.0) * 100.0), 2) if ball_errors else 0.0,
        "ball_detection_rate_pct": round((ball_det_count / max(ball_gt_count, 1)) * 100.0, 2),
        "paddle_responsiveness": {
            "num_counterfactual_states": int(n_cf),
            "up_accuracy_pct": round((up_ok / max(n_cf, 1)) * 100.0, 2),
            "down_accuracy_pct": round((down_ok / max(n_cf, 1)) * 100.0, 2),
            "none_accuracy_pct": round((none_ok / max(n_cf, 1)) * 100.0, 2),
            "strict_ordering_pct": round((strict_order_ok / max(n_cf, 1)) * 100.0, 2),
            "overall_accuracy_pct": round(((up_ok + down_ok + none_ok) / max(3 * n_cf, 1)) * 100.0, 2),
            "mean_delta_y_up_px": round(float(np.mean(up_deltas)), 3) if up_deltas else None,
            "mean_delta_y_none_px": round(float(np.mean(none_deltas)), 3) if none_deltas else None,
            "mean_delta_y_down_px": round(float(np.mean(down_deltas)), 3) if down_deltas else None,
        },
    }


def _generate_eval_episodes(
    num_episodes: int = 20,
    horizon: int = 60,
    seed: int = 77777,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate held-out deterministic ground-truth Pong episodes for rollout comparison.

    Returns:
        init_stacks: (E, 4, 84, 84) float32 in [0, 1]
        actions: (E, horizon) int64
        gt_frames: (E, horizon, 84, 84) float32 in [0, 1]
    """
    rng = np.random.default_rng(seed)
    init_stacks = np.zeros((num_episodes, 4, FRAME_H, FRAME_W), dtype=np.float32)
    actions = np.zeros((num_episodes, horizon), dtype=np.int64)
    gt_frames = np.zeros((num_episodes, horizon, FRAME_H, FRAME_W), dtype=np.float32)

    for ep in range(num_episodes):
        env = PongEnv(seed=seed + ep * 101, ai_jitter=0.0, max_points=99, max_steps=1000)
        # Start cleanly in play with varied trajectories & bounces
        init_state = PongState(
            left_y=int(rng.integers(14, 54)),
            right_y=int(rng.integers(14, 54)),
            ball_x=int(rng.integers(24, 58)),
            ball_y=int(rng.integers(14, 66)),
            ball_vx=2 if (ep % 2 == 0) else -2,
            ball_vy=int(rng.choice([-2, -1, 1, 2])),
        )
        f0 = env.reset(state=init_state)
        init_stacks[ep, 0] = f0.astype(np.float32) * (1.0 / 255.0)
        sticky = {"act": 0, "rem": 0, "offset": 0, "offset_rem": 0, "mode": 0, "mode_rem": 0}
        policy = "tracker" if (ep % 3 != 2) else "mixed"

        for k in range(1, 4):
            a_init = choose_player_action(env, policy=policy, rng=rng, sticky_state=sticky)
            fk, _, _, _ = env.step(a_init)
            init_stacks[ep, k] = fk.astype(np.float32) * (1.0 / 255.0)

        for t in range(horizon):
            a_t = choose_player_action(env, policy=policy, rng=rng, sticky_state=sticky)
            actions[ep, t] = a_t
            ft, _, _, _ = env.step(a_t)
            gt_frames[ep, t] = ft.astype(np.float32) * (1.0 / 255.0)

    return init_stacks, actions, gt_frames


def evaluate_autoregressive_rollout(
    model: torch.nn.Module,
    num_episodes: int = 20,
    horizon: int = 60,
    seed: int = 77777,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Run pure neural 60-step rollouts (ZERO game engine!) and compare against GT."""
    init_stacks, actions, gt_frames = _generate_eval_episodes(
        num_episodes=num_episodes, horizon=horizon, seed=seed
    )
    pred_t, _probs_t = autoregressive_rollout(
        model=model,
        initial_frames=torch.from_numpy(init_stacks),
        actions=torch.from_numpy(actions),
        threshold=0.5,
        use_amp=True,
    )
    pred_frames = pred_t.numpy()  # (E, H, 84, 84)

    curve_full_iou: list[float] = []
    curve_full_f1: list[float] = []
    curve_dyn_iou: list[float] = []
    curve_dyn_f1: list[float] = []
    curve_ball_err: list[float] = []
    curve_ball_det: list[float] = []
    curve_rpad_err: list[float] = []
    curve_lpad_err: list[float] = []

    ep_dyn_scores = np.zeros(num_episodes, dtype=np.float64)
    ep_ball_alive = np.zeros(num_episodes, dtype=np.float64)

    for t in range(horizon):
        f_ious, f_f1s, d_ious, d_f1s = [], [], [], []
        b_errs, r_errs, l_errs = [], [], []
        b_gt = b_det = 0
        for ep in range(num_episodes):
            p = pred_frames[ep, t]
            g = gt_frames[ep, t]
            fm = binary_overlap_metrics(p, g)
            dm = binary_overlap_metrics(p, g, mask=DYNAMIC_MASK)
            f_ious.append(fm["iou"])
            f_f1s.append(fm["f1"])
            d_ious.append(dm["iou"])
            d_f1s.append(dm["f1"])
            ep_dyn_scores[ep] += dm["iou"]

            gb = extract_ball_centroid(g)
            if gb is not None:
                b_gt += 1
                pb = extract_ball_centroid(p)
                if pb is not None:
                    b_det += 1
                    ep_ball_alive[ep] += 1.0
                    b_errs.append(math.hypot(pb[0] - gb[0], pb[1] - gb[1]))

            gr = extract_paddle_cy(g, "right")
            pr = extract_paddle_cy(p, "right")
            if gr is not None and pr is not None:
                r_errs.append(abs(pr - gr))

            gl = extract_paddle_cy(g, "left")
            pl = extract_paddle_cy(p, "left")
            if gl is not None and pl is not None:
                l_errs.append(abs(pl - gl))

        curve_full_iou.append(round(float(np.mean(f_ious)), 5))
        curve_full_f1.append(round(float(np.mean(f_f1s)), 5))
        curve_dyn_iou.append(round(float(np.mean(d_ious)), 5))
        curve_dyn_f1.append(round(float(np.mean(d_f1s)), 5))
        curve_ball_err.append(round(float(np.mean(b_errs)), 4) if b_errs else float("nan"))
        curve_ball_det.append(round((b_det / max(b_gt, 1)) * 100.0, 2))
        curve_rpad_err.append(round(float(np.mean(r_errs)), 4) if r_errs else float("nan"))
        curve_lpad_err.append(round(float(np.mean(l_errs)), 4) if l_errs else float("nan"))

    def _horizon_summary(h: int) -> dict[str, Any]:
        idx = min(h, horizon) - 1
        return {
            "at_step": h,
            "full_frame_iou_at_step": curve_full_iou[idx],
            "full_frame_iou_mean_1_to_step": round(float(np.mean(curve_full_iou[:h])), 5),
            "dynamic_iou_at_step": curve_dyn_iou[idx],
            "dynamic_iou_mean_1_to_step": round(float(np.mean(curve_dyn_iou[:h])), 5),
            "ball_error_px_at_step": curve_ball_err[idx],
            "ball_error_px_mean_1_to_step": round(float(np.nanmean(curve_ball_err[:h])), 4),
            "ball_detection_pct_at_step": curve_ball_det[idx],
            "right_paddle_error_px_at_step": curve_rpad_err[idx],
            "left_paddle_error_px_at_step": curve_lpad_err[idx],
        }

    # Choose a visual showcase episode that has sustained ball play + solid dynamic IoU
    # (quantitative metrics above are ALWAYS the unweighted mean across ALL episodes)
    combined_vis = (ep_dyn_scores / max(horizon, 1)) + 0.6 * (ep_ball_alive / max(horizon, 1))
    showcase_ep = int(np.argmax(combined_vis))

    summary = {
        "num_episodes": int(num_episodes),
        "horizon": int(horizon),
        "horizons": {
            "10_frames": _horizon_summary(10),
            "30_frames": _horizon_summary(30),
            "60_frames": _horizon_summary(60),
        },
        "curves": {
            "steps": list(range(1, horizon + 1)),
            "full_frame_iou": curve_full_iou,
            "full_frame_f1": curve_full_f1,
            "dynamic_iou": curve_dyn_iou,
            "dynamic_f1": curve_dyn_f1,
            "ball_error_px": curve_ball_err,
            "ball_detection_pct": curve_ball_det,
            "right_paddle_error_px": curve_rpad_err,
            "left_paddle_error_px": curve_lpad_err,
        },
        "showcase_episode_idx": showcase_ep,
    }
    return summary, gt_frames[showcase_ep], pred_frames[showcase_ep], actions[showcase_ep]


@torch.no_grad()
def benchmark_inference_speed(model: torch.nn.Module, iters: int = 200) -> dict[str, Any]:
    """Measure real interactive single-frame autoregressive FPS & VRAM."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.randint(0, 2, (1, 4, FRAME_H, FRAME_W), device=device, dtype=torch.float32)
    a = torch.tensor([1], dtype=torch.long, device=device)

    amp_on = device.type == "cuda"
    if amp_on:
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(25):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on):
            nb, _ = model.predict_next_frame(x, a)
        x = torch.cat([x[:, 1:], nb.float()], dim=1)
    if amp_on:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on):
            nb, _ = model.predict_next_frame(x, a)
        x = torch.cat([x[:, 1:], nb.float()], dim=1)
    if amp_on:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms_per_frame = (dt / iters) * 1000.0
    fps = iters / max(dt, 1e-9)
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 1) if amp_on else 0.0
    )
    return {
        "device": str(device),
        "amp_fp16": amp_on,
        "iters": iters,
        "latency_ms_per_frame": round(ms_per_frame, 3),
        "fps": round(fps, 1),
        "peak_inference_vram_mib": peak_vram_mb,
    }


def _colorize_diff(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """RGB diff image: white=match, green=GT only (missed), red=Neural only (extra)."""
    g = gt > 0.5
    p = pred > 0.5
    rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    rgb[:, :] = (10, 14, 24)
    rgb[g & p] = (230, 240, 250)  # exact match -> crisp white
    rgb[g & (~p)] = (34, 197, 94)  # GT only -> green
    rgb[(~g) & p] = (239, 68, 68)  # Neural extra -> red
    return rgb


def _tint_binary(frame: np.ndarray, fg_rgb: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    rgb[:, :] = (8, 12, 20)
    rgb[frame > 0.5] = fg_rgb
    return rgb


def save_rollout_gif(
    gt_seq: np.ndarray,
    pred_seq: np.ndarray,
    actions: np.ndarray,
    out_path: Path,
    scale: int = 3,
    fps: int = 12,
) -> None:
    """Save crisp upscaled side-by-side Ground Truth | Pure Neural | Diff animated GIF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w = FRAME_W * scale
    panel_h = FRAME_H * scale
    gap = 12
    header_h = 34
    canvas_w = panel_w * 3 + gap * 4
    canvas_h = header_h + panel_h + 14

    gif_frames: list[np.ndarray] = []
    for t in range(len(gt_seq)):
        gt_rgb = _tint_binary(gt_seq[t], (255, 232, 170))
        pr_rgb = _tint_binary(pred_seq[t], (180, 255, 220))
        df_rgb = _colorize_diff(gt_seq[t], pred_seq[t])

        dyn_iou = binary_overlap_metrics(pred_seq[t], gt_seq[t], mask=DYNAMIC_MASK)["iou"]
        act_name = ACTION_NAMES[int(actions[t])]

        canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 18, 32))
        draw = ImageDraw.Draw(canvas)

        for col_idx, (arr, title, color) in enumerate(
            [
                (gt_rgb, "GROUND TRUTH ENGINE", (251, 191, 36)),
                (pr_rgb, "PURE NEURAL MODEL (0 CODE)", (52, 211, 153)),
                (df_rgb, "DIFF (W=OK G=MISS R=EXTRA)", (248, 113, 113)),
            ]
        ):
            img_p = Image.fromarray(arr).resize((panel_w, panel_h), resample=Image.NEAREST)
            x0 = gap + col_idx * (panel_w + gap)
            canvas.paste(img_p, (x0, header_h))
            draw.text((x0 + 4, 18), title, fill=color)

        draw.text(
            (gap, 3),
            f"Autoregressive Step t={t + 1:02d}/{len(gt_seq)}   |   Action: {act_name:<4s}   |   Dynamic IoU: {dyn_iou:.3f}",
            fill=(226, 232, 240),
        )
        gif_frames.append(np.asarray(canvas))

    imageio.mimsave(out_path, gif_frames, duration=1.0 / fps, loop=0)


def save_rollout_grid(
    gt_seq: np.ndarray,
    pred_seq: np.ndarray,
    rollout_summary: dict[str, Any],
    tf_metrics: dict[str, Any],
    out_path: Path,
) -> None:
    """Save publication-grade PNG comparing GT vs Neural keyframes + 60-step decay curves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key_steps = [1, 5, 10, 20, 30, 45, 60]
    key_steps = [s for s in key_steps if s <= len(gt_seq)]
    n_cols = len(key_steps)

    fig = plt.figure(figsize=(2.4 * n_cols, 9.5), facecolor="#0b1120")
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[3.15, 1.55], hspace=0.22)
    gs_top = gs_outer[0].subgridspec(3, n_cols, hspace=0.14, wspace=0.12)
    gs_bot = gs_outer[1].subgridspec(1, 2, wspace=0.24)

    fig.suptitle(
        "Neural Pong World Model — Pure Autoregressive Rollout (ZERO Game Engine)\n"
        f"1-Step Dynamic IoU={tf_metrics['dynamic_region_iou']:.3f} | "
        f"Ball Err={tf_metrics['ball_centroid_error_px_mean']:.2f} px | "
        f"Paddle Resp={tf_metrics['paddle_responsiveness']['overall_accuracy_pct']:.1f}%",
        color="#f8fafc",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )

    row_labels = [
        ("Ground Truth\n(Pong Engine)", "#fbbf24"),
        ("Pure Neural\nWorld Model", "#34d399"),
        ("Overlay Diff\n(W=OK,G=GT,R=Pred)", "#f87171"),
    ]

    for c, step_num in enumerate(key_steps):
        idx = step_num - 1
        gt_img = _tint_binary(gt_seq[idx], (255, 232, 170))
        pr_img = _tint_binary(pred_seq[idx], (180, 255, 220))
        df_img = _colorize_diff(gt_seq[idx], pred_seq[idx])

        for r, img in enumerate([gt_img, pr_img, df_img]):
            ax = fig.add_subplot(gs_top[r, c])
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#334155")
                sp.set_linewidth(1.0)
            if r == 0:
                ax.set_title(f"t = {step_num}", color="#e2e8f0", fontsize=10, pad=5)
            if c == 0:
                ax.set_ylabel(row_labels[r][0], color=row_labels[r][1], fontsize=9, fontweight="bold")

    # Bottom plots: IoU decay & centroid errors over 60 steps
    curves = rollout_summary["curves"]
    steps = curves["steps"]

    ax_iou = fig.add_subplot(gs_bot[0, 0])
    ax_iou.set_facecolor("#0f172a")
    ax_iou.plot(steps, curves["full_frame_iou"], label="Full-Frame IoU", color="#38bdf8", lw=2.2)
    ax_iou.plot(steps, curves["dynamic_iou"], label="Dynamic IoU (Ball + Paddles)", color="#34d399", lw=2.2)
    ax_iou.axvline(10, color="#64748b", ls="--", lw=1)
    ax_iou.axvline(30, color="#64748b", ls="--", lw=1)
    ax_iou.set_ylim(0.0, 1.05)
    ax_iou.set_xlim(1, len(steps))
    ax_iou.set_xlabel("Autoregressive Step (ZERO Game Engine)", color="#cbd5e1", fontsize=9)
    ax_iou.set_ylabel("IoU", color="#cbd5e1", fontsize=9)
    ax_iou.set_title("Multi-Step Rollout IoU Decay (20 Held-Out Episodes)", color="#f8fafc", fontsize=10.5)
    ax_iou.tick_params(colors="#94a3b8")
    ax_iou.grid(color="#1e293b", lw=0.7)
    ax_iou.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    ax_err = fig.add_subplot(gs_bot[0, 1])
    ax_err.set_facecolor("#0f172a")
    ax_err.plot(steps, curves["ball_error_px"], label="Ball Centroid Error (px)", color="#fb923c", lw=2.2)
    ax_err.plot(steps, curves["right_paddle_error_px"], label="Right Paddle Error (px)", color="#a78bfa", lw=2.0)
    ax_err.plot(steps, curves["left_paddle_error_px"], label="Left AI Paddle Error (px)", color="#f472b6", lw=1.6, ls="--")
    ax_err.set_xlim(1, len(steps))
    ax_err.set_ylim(bottom=0.0)
    ax_err.set_xlabel("Autoregressive Step (ZERO Game Engine)", color="#cbd5e1", fontsize=9)
    ax_err.set_ylabel("Centroid Error (px)", color="#cbd5e1", fontsize=9)
    ax_err.set_title("Ball & Paddle Spatial Drift Over 60 Steps", color="#f8fafc", fontsize=10.5)
    ax_err.tick_params(colors="#94a3b8")
    ax_err.grid(color="#1e293b", lw=0.7)
    ax_err.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_full_evaluation(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    reports_dir: str | Path,
    num_tf_samples: int = 2000,
    num_rollout_episodes: int = 20,
    rollout_horizon: int = 60,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute complete evaluation suite and write JSON + GIF + PNG reports."""
    data_p = Path(data_dir)
    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)

    with open(data_p / "dataset_meta.json", "r", encoding="utf-8") as f:
        ds_meta = json.load(f)
    test_bounds = tuple(ds_meta["splits"]["test"])

    model, model_meta = load_world_model(checkpoint_path, device=device)
    dev = next(model.parameters()).device

    tf_metrics = evaluate_teacher_forced(
        model=model,
        data_dir=data_p,
        test_bounds=(int(test_bounds[0]), int(test_bounds[1])),
        max_samples=num_tf_samples,
        device=dev,
    )
    rollout_summary, gt_show, pred_show, act_show = evaluate_autoregressive_rollout(
        model=model,
        num_episodes=num_rollout_episodes,
        horizon=rollout_horizon,
    )
    speed_bench = benchmark_inference_speed(model, iters=200)

    gif_path = rep_p / "rollout_comparison.gif"
    grid_path = rep_p / "rollout_grid.png"
    save_rollout_gif(gt_show, pred_show, act_show, gif_path, scale=3, fps=12)
    save_rollout_grid(gt_show, pred_show, rollout_summary, tf_metrics, grid_path)

    report = {
        "model": model_meta,
        "inference_benchmark": speed_bench,
        "teacher_forced_1step": tf_metrics,
        "autoregressive_rollout": rollout_summary,
        "artifacts": {
            "eval_json": str(rep_p / "eval_rollout.json"),
            "comparison_gif": str(gif_path),
            "comparison_grid_png": str(grid_path),
        },
    }
    with open(rep_p / "eval_rollout.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if dev.type == "cuda":
        del model
        torch.cuda.empty_cache()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Neural Pong World Model.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/neural_pong_best.pt")
    parser.add_argument("--data-dir", type=str, default="data/pong_140k")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--tf-samples", type=int, default=2000)
    parser.add_argument("--rollout-episodes", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=60)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    report = run_full_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        num_tf_samples=args.tf_samples,
        num_rollout_episodes=args.rollout_episodes,
        rollout_horizon=args.rollout_horizon,
        device=args.device,
    )
    print(json.dumps({
        "model": report["model"],
        "inference_benchmark": report["inference_benchmark"],
        "teacher_forced_1step": report["teacher_forced_1step"],
        "autoregressive_horizons": report["autoregressive_rollout"]["horizons"],
    }, indent=2))


if __name__ == "__main__":
    main()
