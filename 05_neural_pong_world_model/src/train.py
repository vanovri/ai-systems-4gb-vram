"""Training pipeline for Action-Conditioned Neural Pong World Model (GTX 1650 Ti FP16 AMP).

Key training innovations:
1. Strict FP16 AMP (`torch.amp.autocast('cuda', dtype=torch.float16)` + `GradScaler`)
   with loss computed in float32 directly on raw logits (fixing Slide #6's gradient bug).
2. `BCEWithLogits(pos_weight) + Motion-Weighted BCE + Soft Dice Loss` preventing
   all-black collapse and keeping the 3x3 ball razor sharp.
3. GameNGen-style input corruption augmentation (random bit-flips, sparse foreground
   dropout, mild additive jitter) during Phase 1 teacher-forced training.
4. Phase 2 Multi-Step Autoregressive Unrolled Fine-Tuning (K=4 steps with Straight-Through
   Estimator binarization `ste_binarize`) so the model learns to stay drift-free when
   feeding its own binary predictions back into the 4-frame context window.
5. Per-epoch checkpointing in `artifacts/checkpoints/` + best checkpoint selection by
   real autoregressive rollout Dynamic IoU!
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import PongSequenceDataset
from src.eval_rollout import DYNAMIC_MASK, binary_overlap_metrics, extract_ball_centroid
from src.model import (
    ModelSize,
    NeuralPongWorldModel,
    configure_cuda_backend,
    ste_binarize,
    world_model_loss,
)
from src.neural_runner import autoregressive_rollout


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_val_quick(
    model: NeuralPongWorldModel,
    val_1step_ds: PongSequenceDataset,
    val_rollout_ds: PongSequenceDataset,
    device: torch.device,
    max_1step: int = 512,
    num_rollouts: int = 16,
    rollout_horizon: int = 15,
) -> dict[str, float]:
    """Fast validation: 1-step BCE/Dynamic-IoU + 15-step pure neural rollout Dynamic-IoU."""
    model.eval()
    amp_on = device.type == "cuda"

    # 1-step val
    n1 = min(len(val_1step_ds), max_1step)
    ids1 = np.linspace(0, len(val_1step_ds) - 1, n1, dtype=np.int64)
    bce_sum = 0.0
    dyn_ious: list[float] = []
    full_ious: list[float] = []
    ball_errs: list[float] = []

    for s in range(0, n1, 64):
        chunk = ids1[s : s + 64]
        xs, acts, ys = zip(*(val_1step_ds[int(i)] for i in chunk))
        x_t = torch.stack(xs).to(device)
        a_t = torch.stack(acts).to(device)
        y_t = torch.stack(ys).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on):
            logits = model(x_t, a_t)
        _, m = world_model_loss(logits, y_t, last_frame=x_t[:, -1:])
        bce_sum += m["raw_bce"] * len(chunk)

        pred_np = (torch.sigmoid(logits.float())[:, 0].cpu().numpy() >= 0.5).astype(np.float32)
        gt_np = y_t[:, 0].cpu().numpy()
        for p, g in zip(pred_np, gt_np):
            full_ious.append(binary_overlap_metrics(p, g)["iou"])
            dyn_ious.append(binary_overlap_metrics(p, g, mask=DYNAMIC_MASK)["iou"])
            gb = extract_ball_centroid(g)
            if gb is not None:
                pb = extract_ball_centroid(p)
                if pb is not None:
                    ball_errs.append(math.hypot(pb[0] - gb[0], pb[1] - gb[1]))

    # 15-step pure neural rollout val (on held-out val episodes!)
    nr = min(len(val_rollout_ds), num_rollouts)
    rollout_dyn_iou = 0.0
    if nr > 0:
        ids_r = np.linspace(0, len(val_rollout_ds) - 1, nr, dtype=np.int64)
        xs_r, acts_r, ys_r = zip(*(val_rollout_ds[int(i)] for i in ids_r))
        init_t = torch.stack(xs_r)
        act_t = torch.stack(acts_r)[:, :rollout_horizon]
        gt_r = torch.stack(ys_r)[:, :rollout_horizon].numpy()

        pred_r, _ = autoregressive_rollout(
            model=model,
            initial_frames=init_t,
            actions=act_t,
            threshold=0.5,
            use_amp=amp_on,
        )
        pred_r_np = pred_r.numpy()
        r_ious = [
            binary_overlap_metrics(pred_r_np[e, t], gt_r[e, t], mask=DYNAMIC_MASK)["iou"]
            for e in range(nr)
            for t in range(rollout_horizon)
        ]
        rollout_dyn_iou = float(np.mean(r_ious))

    raw_bce = bce_sum / max(n1, 1)
    return {
        "val_raw_bce": round(raw_bce, 6),
        "val_pseudo_ppl": round(math.exp(min(raw_bce, 20.0)), 6),
        "val_full_iou": round(float(np.mean(full_ious)), 5),
        "val_dynamic_iou": round(float(np.mean(dyn_ious)), 5),
        "val_ball_err_px": round(float(np.mean(ball_errs)), 3) if ball_errs else 99.0,
        "val_rollout15_dynamic_iou": round(rollout_dyn_iou, 5),
    }


def _sample_epoch_loader(
    dataset: PongSequenceDataset,
    batch_size: int,
    max_samples: int | None,
    rng: np.random.Generator,
) -> DataLoader:
    n = len(dataset)
    if max_samples is not None and max_samples < n:
        chosen = rng.choice(n, size=max_samples, replace=False)
        sub = Subset(dataset, chosen.tolist())
        return DataLoader(
            sub,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def train_world_model(
    data_dir: str | Path = "data/pong_140k",
    artifacts_dir: str | Path = "artifacts",
    model_size: ModelSize = "small",
    phase1_epochs: int = 7,
    phase2_epochs: int = 3,
    batch_size_p1: int = 64,
    batch_size_p2: int = 32,
    samples_per_epoch_p1: int = 9_600,
    samples_per_epoch_p2: int = 1_600,
    unroll_steps: int = 4,
    lr_p1: float = 1.5e-3,
    lr_p2: float = 3.0e-4,
    weight_decay: float = 1e-4,
    pos_weight_p1: float = 5.0,
    pos_weight_p2: float = 2.0,
    dice_weight: float = 1.0,
    motion_weight: float = 6.0,
    seed: int = 2026,
    resume_path: str | Path | None = None,
    detach_rollout: bool = True,
    reset_optimizer: bool = False,
) -> dict[str, Any]:
    """Train NeuralPongWorldModel on GTX 1650 Ti with FP16 AMP + 2-phase anti-drift schedule."""
    set_seed(seed)
    cuda_info = configure_cuda_backend()
    data_p = Path(data_dir)
    art_p = Path(artifacts_dir)
    ckpt_dir = art_p / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(data_p / "dataset_meta.json", "r", encoding="utf-8") as f:
        ds_meta = json.load(f)
    train_bounds = tuple(ds_meta["splits"]["train"])
    val_bounds = tuple(ds_meta["splits"]["val"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = device.type == "cuda"
    if amp_on:
        torch.cuda.reset_peak_memory_stats()

    model = NeuralPongWorldModel(model_size).to(device)
    param_count = model.count_parameters()
    print(
        f"[Train] Device={device} ({cuda_info['gpu_name']}) | AMP_FP16={amp_on} | "
        f"cuDNN={cuda_info['cudnn_enabled']} | Model={model_size} ({param_count:,} params)",
        flush=True,
    )

    train_1step_ds = PongSequenceDataset(
        data_dir=data_p,
        start_idx=int(train_bounds[0]),
        end_idx=int(train_bounds[1]),
        stack_size=4,
        rollout_steps=1,
        augment_noise=True,
        bitflip_prob_max=0.004,
        drop_pixel_prob=0.03,
    )
    train_unroll_ds = PongSequenceDataset(
        data_dir=data_p,
        start_idx=int(train_bounds[0]),
        end_idx=int(train_bounds[1]),
        stack_size=4,
        rollout_steps=unroll_steps,
        augment_noise=True,
        bitflip_prob_max=0.0025,
        drop_pixel_prob=0.02,
    )
    val_1step_ds = PongSequenceDataset(
        data_dir=data_p,
        start_idx=int(val_bounds[0]),
        end_idx=int(val_bounds[1]),
        stack_size=4,
        rollout_steps=1,
        augment_noise=False,
    )
    val_rollout_ds = PongSequenceDataset(
        data_dir=data_p,
        start_idx=int(val_bounds[0]),
        end_idx=int(val_bounds[1]),
        stack_size=4,
        rollout_steps=15,
        augment_noise=False,
    )
    print(
        f"[Train] Train 1-step={len(train_1step_ds):,} | Train {unroll_steps}-step={len(train_unroll_ds):,} | "
        f"Val 1-step={len(val_1step_ds):,}",
        flush=True,
    )

    total_epochs = phase1_epochs + phase2_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_p1, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on, init_scale=4096.0)

    start_epoch = 1
    best_score = -1.0
    history: list[dict[str, Any]] = []

    if resume_path is not None and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if (not reset_optimizer) and ("optimizer_state_dict" in ckpt):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        history = ckpt.get("history", [])
        print(
            f"[Train] Resumed from {resume_path} at epoch {start_epoch} "
            f"(reset_opt={reset_optimizer}, detach_rollout={detach_rollout})",
            flush=True,
        )

    rng = np.random.default_rng(seed)
    t_train_start = time.perf_counter()

    # Step weights for multi-step unrolled training (later steps get slightly higher weight)
    step_weights = [1.0 + 0.15 * k for k in range(unroll_steps)]
    step_weight_sum = sum(step_weights)

    for epoch in range(start_epoch, total_epochs + 1):
        is_phase2 = epoch > phase1_epochs
        phase_name = f"P2-Unroll{unroll_steps}" if is_phase2 else "P1-1Step"
        t_ep = time.perf_counter()

        # Cosine LR schedule within each phase
        if not is_phase2:
            progress = (epoch - 1) / max(phase1_epochs, 1)
            lr_now = lr_p1 * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            loader = _sample_epoch_loader(train_1step_ds, batch_size_p1, samples_per_epoch_p1, rng)
        else:
            p2_idx = epoch - phase1_epochs - 1
            progress = p2_idx / max(phase2_epochs, 1)
            lr_now = lr_p2 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            loader = _sample_epoch_loader(train_unroll_ds, batch_size_p2, samples_per_epoch_p2, rng)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.train()
        loss_sum = 0.0
        raw_bce_sum = 0.0
        dice_sum = 0.0
        num_batches = 0

        for x, act, y in loader:
            x = x.to(device, non_blocking=True)
            act = act.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if not is_phase2:
                # Phase 1: 1-step teacher-forced with noise augmentation
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on):
                    logits = model(x, act)
                    loss, m = world_model_loss(
                        logits=logits,
                        target=y,
                        last_frame=x[:, -1:],
                        pos_weight=pos_weight_p1,
                        dice_weight=dice_weight,
                        motion_weight=motion_weight,
                    )
                raw_bce_b = m["raw_bce"]
                dice_b = m["dice_loss"]
            else:
                # Phase 2: K-step autoregressive unrolled training with STE binary feedback
                buf = x
                total_loss = torch.zeros((), device=device, dtype=torch.float32)
                raw_bce_acc = 0.0
                dice_acc = 0.0
                for k in range(unroll_steps):
                    target_k = y[:, k : k + 1]
                    action_k = act[:, k]
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on):
                        logits_k = model(buf, action_k)
                        loss_k, mk = world_model_loss(
                            logits=logits_k,
                            target=target_k,
                            last_frame=buf[:, -1:],
                            pos_weight=pos_weight_p2,
                            dice_weight=dice_weight,
                            motion_weight=motion_weight,
                        )
                    total_loss = total_loss + (step_weights[k] / step_weight_sum) * loss_k
                    raw_bce_acc += mk["raw_bce"]
                    dice_acc += mk["dice_loss"]
                    if k + 1 < unroll_steps:
                        if detach_rollout:
                            # DAgger / Scheduled-Sampling binary feedback (exact inference
                            # distribution without noisy multi-step STE BPTT shock!)
                            next_fb = (logits_k.detach().float() >= 0.0).to(dtype=buf.dtype)
                        else:
                            # Straight-Through binary feedback
                            next_fb = ste_binarize(logits_k.float())
                        buf = torch.cat([buf[:, 1:, :, :], next_fb], dim=1)
                loss = total_loss
                raw_bce_b = raw_bce_acc / unroll_steps
                dice_b = dice_acc / unroll_steps

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss encountered at epoch {epoch}, batch {num_batches} ({phase_name})!"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach().item())
            raw_bce_sum += raw_bce_b
            dice_sum += dice_b
            num_batches += 1

        val_metrics = evaluate_val_quick(
            model=model,
            val_1step_ds=val_1step_ds,
            val_rollout_ds=val_rollout_ds,
            device=device,
        )
        ep_sec = time.perf_counter() - t_ep
        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated() / (1024**2), 1) if amp_on else 0.0
        )

        # Selection metric prioritizes autoregressive stability + 1-step crispness
        score = 0.65 * val_metrics["val_rollout15_dynamic_iou"] + 0.35 * val_metrics["val_dynamic_iou"]
        is_best = score > best_score
        if is_best:
            best_score = score

        rec = {
            "epoch": epoch,
            "phase": phase_name,
            "lr": round(lr_now, 6),
            "train_loss": round(loss_sum / max(num_batches, 1), 5),
            "train_raw_bce": round(raw_bce_sum / max(num_batches, 1), 6),
            "train_dice_loss": round(dice_sum / max(num_batches, 1), 5),
            **val_metrics,
            "selection_score": round(score, 5),
            "is_best": is_best,
            "epoch_sec": round(ep_sec, 1),
            "peak_vram_mib": peak_vram_mb,
        }
        history.append(rec)

        print(
            f"[Ep {epoch:02d}/{total_epochs:02d} {phase_name:10s}] "
            f"loss={rec['train_loss']:.4f} | val_bce={rec['val_raw_bce']:.5f} | "
            f"val_dyn_iou={rec['val_dynamic_iou']:.4f} | val_ball_err={rec['val_ball_err_px']:.2f}px | "
            f"val_roll15_iou={rec['val_rollout15_dynamic_iou']:.4f} | "
            f"{ep_sec:.1f}s | VRAM={peak_vram_mb:.0f}MiB {'*BEST*' if is_best else ''}",
            flush=True,
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_size": model_size,
            "param_count": param_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "val_metrics": val_metrics,
            "history": history,
        }
        torch.save(ckpt_payload, ckpt_dir / f"epoch_{epoch:02d}.pt")
        if is_best:
            torch.save(ckpt_payload, art_p / "neural_pong_best.pt")

    total_sec = time.perf_counter() - t_train_start
    summary = {
        "model_size": model_size,
        "param_count": param_count,
        "device": str(device),
        "amp_fp16": amp_on,
        "total_epochs": total_epochs,
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "total_train_sec": round(total_sec, 1),
        "best_selection_score": round(best_score, 5),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1) if amp_on else 0.0,
        "best_checkpoint": str(art_p / "neural_pong_best.pt"),
        "history": history,
    }
    with open(art_p / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if amp_on:
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neural Pong World Model with FP16 AMP.")
    parser.add_argument("--data-dir", type=str, default="data/pong_140k")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--model-size", type=str, choices=["small", "medium"], default="small")
    parser.add_argument("--phase1-epochs", type=int, default=7)
    parser.add_argument("--phase2-epochs", type=int, default=3)
    parser.add_argument("--batch-size-p1", type=int, default=64)
    parser.add_argument("--batch-size-p2", type=int, default=32)
    parser.add_argument("--samples-per-epoch-p1", type=int, default=9_600)
    parser.add_argument("--samples-per-epoch-p2", type=int, default=1_600)
    parser.add_argument("--unroll-steps", type=int, default=4)
    parser.add_argument("--lr-p1", type=float, default=1.5e-3)
    parser.add_argument("--lr-p2", type=float, default=3.0e-4)
    parser.add_argument("--pos-weight-p1", type=float, default=5.0)
    parser.add_argument("--pos-weight-p2", type=float, default=3.2)
    parser.add_argument("--dice-weight", type=float, default=1.2)
    parser.add_argument("--motion-weight", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ste-rollout", action="store_true", help="Use STE BPTT instead of detached DAgger rollout in Phase 2")
    parser.add_argument("--reset-optimizer", action="store_true", help="Reset AdamW moments on resume")
    args = parser.parse_args()

    summary = train_world_model(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        model_size=args.model_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size_p1=args.batch_size_p1,
        batch_size_p2=args.batch_size_p2,
        samples_per_epoch_p1=args.samples_per_epoch_p1,
        samples_per_epoch_p2=args.samples_per_epoch_p2,
        unroll_steps=args.unroll_steps,
        lr_p1=args.lr_p1,
        lr_p2=args.lr_p2,
        pos_weight_p1=args.pos_weight_p1,
        pos_weight_p2=args.pos_weight_p2,
        dice_weight=args.dice_weight,
        motion_weight=args.motion_weight,
        seed=args.seed,
        resume_path=args.resume,
        detach_rollout=not args.ste_rollout,
        reset_optimizer=args.reset_optimizer,
    )
    print(f"[Train] Complete in {summary['total_train_sec']:.1f}s! Best score={summary['best_selection_score']:.4f}", flush=True)


if __name__ == "__main__":
    main()
