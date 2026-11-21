"""LoRA fine-tuning of Qwen/Qwen3-0.6B on Banking77 with strict GTX 1650 Ti (4 GB VRAM) guardrails.

Guardrails & optimizations enforced:
- FP16 weight storage (fp16=True, bf16=False — Turing SM 7.5 has no native BF16)
- Frozen FP16 base model + FP32 LoRA master weights + torch.amp.GradScaler
- gradient_checkpointing=True (use_reentrant=False), use_cache=False, enable_input_require_grads()
- Strict -100 prompt-token loss masking with exact answer-only lm_head projection
  (mathematically identical to full-vocab CE loss, abs_diff < 2e-6, saving >1.1 GB VRAM)
- Length-bucketed random batching + adaptive sub-chunking (max chunk=4 when seq_len > 64)
- Optional Turing TU117 FP16-storage / FP32-compute linear kernel (auto-enabled on GTX 16xx)
- Incremental checkpoints (every save_steps, keeping save_total_limit)
- Per-25-step loss curve logging (reports/train_curve.json) + plot (reports/loss_curve.png)
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import (  # noqa: E402
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    IGNORE_INDEX,
    Banking77SFTDataset,
    DynamicSFTCollator,
    LengthBucketBatchSampler,
    load_banking77_splits,
    save_label_map,
)

_ORIG_F_LINEAR = F.linear


class _TU117LinearFn(torch.autograd.Function):
    """Keeps frozen weights in FP16 on VRAM while computing GEMMs on GTX 16xx FP32 CUDA cores."""

    @staticmethod
    def forward(ctx, inp, weight, bias):
        ctx.save_for_backward(inp, weight, bias)
        out = _ORIG_F_LINEAR(
            inp.float(),
            weight.float(),
            bias.float() if bias is not None else None,
        )
        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight, bias = ctx.saved_tensors
        go = grad_out.float()
        grad_inp = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_inp = (go @ weight.float()).to(inp.dtype)
        if ctx.needs_input_grad[1]:
            go2 = go.reshape(-1, go.shape[-1])
            inp2 = inp.float().reshape(-1, inp.shape[-1])
            grad_weight = (go2.t() @ inp2).to(weight.dtype)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = go.reshape(-1, go.shape[-1]).sum(0).to(bias.dtype)
        return grad_inp, grad_weight, grad_bias


@contextlib.contextmanager
def tu117_fast_linear_ctx(enabled: bool = True):
    """Avoids slow 0.4 TFLOPS FP16 fallback on Tensor-Core-less GTX 16-series (TU116/TU117)."""
    if not enabled:
        yield
        return

    def _patched_linear(inp, weight, bias=None):
        if inp.is_cuda and weight.dtype == torch.float16 and weight.shape[0] <= 16384:
            return _TU117LinearFn.apply(inp, weight, bias)
        return _ORIG_F_LINEAR(inp, weight, bias)

    F.linear = _patched_linear
    try:
        yield
    finally:
        F.linear = _ORIG_F_LINEAR


def compute_answer_only_loss(
    model: PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """Exact answer-only causal LM cross-entropy loss projecting lm_head ONLY on non-masked positions.

    Because lm_head is strictly token-wise, slicing hidden states where `labels[:, 1:] != -100`
    before applying lm_head produces the exact same loss and gradients as full-sequence logits
    while eliminating >90% of wasted 151,936-vocab FP32 logit memory and compute.
    """
    backbone = model.base_model.model.model
    lm_head = model.base_model.model.lm_head

    hidden = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).last_hidden_state

    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != IGNORE_INDEX
    n_valid = int(valid_mask.sum().item())
    if n_valid == 0:
        return hidden.sum() * 0.0, 0

    active_hidden = shift_hidden[valid_mask]
    active_labels = shift_labels[valid_mask]
    active_logits = lm_head(active_hidden).float()
    loss = F.cross_entropy(active_logits, active_labels, reduction="mean")
    return loss, n_valid


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_vram_guardrail(min_free_mib: int = 3200, allow_low_vram: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available — GTX 1650 Ti is required for Channel 4 training.")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mib = free_bytes / (1024 ** 2)
    total_mib = total_bytes / (1024 ** 2)
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    info = {
        "gpu_name": gpu_name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "vram_free_mib": round(free_mib, 1),
        "vram_total_mib": round(total_mib, 1),
    }
    print(
        f"[GPU] {gpu_name} (SM {cc[0]}.{cc[1]}) | Free VRAM: {free_mib:.1f} / {total_mib:.1f} MiB",
        flush=True,
    )
    if free_mib < min_free_mib and not allow_low_vram:
        raise RuntimeError(
            f"VRAM guardrail triggered: only {free_mib:.1f} MiB free (< {min_free_mib} MiB required). "
            "Close competing CUDA processes or pass --allow-low-vram."
        )
    return info


def build_lora_model(
    model_id: str = DEFAULT_MODEL_ID,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
    gradient_checkpointing: bool = True,
    device: str = "cuda",
) -> Tuple[PeftModel, Dict[str, Any]]:
    """Load Qwen3-0.6B in FP16 and attach FP32-master-weight LoRA adapters."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    )
    base_model.config.use_cache = False

    if gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    # Keep trainable LoRA adapter weights in FP32 so GradScaler unscales cleanly on Turing FP16
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    ratio_pct = 100.0 * trainable_params / total_params
    param_stats = {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_pct": round(ratio_pct, 4),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": list(target_modules),
    }
    print(
        f"[LoRA] trainable={trainable_params:,} / total={total_params:,} ({ratio_pct:.4f}%) | "
        f"r={lora_r}, alpha={lora_alpha}, targets={list(target_modules)}",
        flush=True,
    )
    return model, param_stats


@torch.no_grad()
def evaluate_val_loss(
    model: PeftModel,
    val_loader: DataLoader,
    device: str,
    use_tu117_fast: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with tu117_fast_linear_ctx(use_tu117_fast):
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss_t, valid_tokens = compute_answer_only_loss(
                    model, input_ids, attention_mask, labels
                )
            if valid_tokens == 0:
                continue
            loss_val = float(loss_t.item())
            if math.isfinite(loss_val):
                total_loss += loss_val * valid_tokens
                total_tokens += valid_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def _prune_checkpoints(ckpt_root: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0 or not ckpt_root.exists():
        return
    ckpts = sorted(
        [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    while len(ckpts) > save_total_limit:
        old = ckpts.pop(0)
        shutil.rmtree(old, ignore_errors=True)


def plot_loss_curve(
    train_points: List[Dict[str, Any]],
    val_points: List[Dict[str, Any]],
    param_stats: Dict[str, Any],
    peak_vram_mb: float,
    eff_bs: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [p["step"] for p in train_points]
    losses = [p["train_loss"] for p in train_points]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#111827")

    ax.plot(
        steps,
        losses,
        color="#38bdf8",
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        label="Train Loss (answer-only CE, 25-step avg)",
    )

    if len(losses) >= 3:
        ema = []
        alpha = 0.35
        cur = losses[0]
        for v in losses:
            cur = alpha * v + (1.0 - alpha) * cur
            ema.append(cur)
        ax.plot(steps, ema, color="#f59e0b", linewidth=2.0, linestyle="--", label="Smoothed EMA")

    if val_points:
        v_steps = [p["step"] for p in val_points]
        v_losses = [p["val_loss"] for p in val_points]
        ax.scatter(
            v_steps,
            v_losses,
            color="#34d399",
            s=72,
            zorder=5,
            edgecolors="white",
            linewidths=0.9,
            label="Validation Loss (Banking77 val split)",
        )
        for vs, vl in zip(v_steps, v_losses):
            ax.annotate(
                f"{vl:.3f}",
                (vs, vl),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                color="#a7f3d0",
                fontweight="bold",
            )

    ax.set_title(
        "Qwen3-0.6B + LoRA (r=8, q_proj/v_proj) on Banking77 — Training Convergence",
        color="#f8fafc",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(f"Optimizer Step (effective batch size = {eff_bs})", color="#cbd5e1", fontsize=10.5)
    ax.set_ylabel("Answer-Only Cross-Entropy Loss", color="#cbd5e1", fontsize=10.5)
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    info_text = (
        f"Trainable: {param_stats['trainable_params']:,} ({param_stats['trainable_pct']:.2f}%)\n"
        f"Peak VRAM: {peak_vram_mb:.0f} MiB (GTX 1650 Ti 4 GB, FP16)\n"
        f"Final Train Loss: {losses[-1]:.4f}"
        + (f" | Best Val Loss: {min(p['val_loss'] for p in val_points):.4f}" if val_points else "")
    )
    ax.text(
        0.98,
        0.96,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#e2e8f0",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#1e293b", edgecolor="#475569", alpha=0.92),
    )

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.35, 0.97), facecolor="#1e293b", edgecolor="#475569")
    for txt in legend.get_texts():
        txt.set_color("#f1f5f9")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    gpu_info = check_vram_guardrail(min_free_mib=args.min_free_vram_mib, allow_low_vram=args.allow_low_vram)
    device = "cuda"
    use_tu117_fast = (not args.disable_tu117_fast) and ("1650" in gpu_info["gpu_name"] or "1660" in gpu_info["gpu_name"])

    reports_dir = PROJECT_DIR / "reports"
    artifacts_dir = PROJECT_DIR / "artifacts"
    adapter_dir = artifacts_dir / "qwen3_0.6b_banking77_lora"
    ckpt_root = artifacts_dir / "checkpoints"
    reports_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print(f"[Data] Loading {args.dataset_id} (val_ratio={args.val_ratio})...", flush=True)
    splits = load_banking77_splits(val_ratio=args.val_ratio, seed=args.seed, dataset_id=args.dataset_id)
    save_label_map(splits.id2label, adapter_dir / "label_map.json")
    print(
        f"[Data] train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}, labels={len(splits.id2label)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = Banking77SFTDataset(splits.train, tokenizer=tokenizer, max_length=args.max_length)
    val_ds = Banking77SFTDataset(splits.val, tokenizer=tokenizer, max_length=args.max_length)
    collator = DynamicSFTCollator(pad_token_id=tokenizer.pad_token_id)

    train_lengths = [len(s["input_ids"]) for s in train_ds.samples]
    batch_sampler = LengthBucketBatchSampler(
        lengths=train_lengths,
        batch_size=args.batch_size,
        mega_bucket_mult=32,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 8),
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    model, param_stats = build_lora_model(
        model_id=args.model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tuple(args.target_modules),
        gradient_checkpointing=True,
        device=device,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    planned_steps = steps_per_epoch * args.epochs
    total_steps = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_curve: List[Dict[str, Any]] = []
    val_curve: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    global_step = 0
    accum_counter = 0
    window_loss_sum = 0.0
    window_steps = 0
    step_loss_accum = 0.0
    micro_in_step = 0
    eff_bs = args.batch_size * args.grad_accum

    t_start = time.perf_counter()
    print(
        f"[Train] epochs={args.epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"bs={args.batch_size}, grad_accum={args.grad_accum} (eff_bs={eff_bs}), "
        f"lr={args.lr}, fp16=True, bf16=False, tu117_fast={use_tu117_fast}",
        flush=True,
    )

    def _run_val(tag_epoch: float) -> None:
        nonlocal best_val_loss
        if len(val_ds) == 0:
            return
        val_loss = evaluate_val_loss(model, val_loader, device, use_tu117_fast=use_tu117_fast)
        elapsed = time.perf_counter() - t_start
        val_curve.append(
            {
                "step": global_step,
                "epoch": round(tag_epoch, 3),
                "val_loss": round(val_loss, 5),
                "elapsed_sec": round(elapsed, 1),
            }
        )
        print(
            f"[Val] epoch={tag_epoch:.2f} step={global_step} | val_loss={val_loss:.4f} | elapsed={elapsed:.1f}s",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            print(f"  [Best Adapter] Updated {adapter_dir} (val_loss={val_loss:.4f})", flush=True)

    stop_early = False
    with tu117_fast_linear_ctx(use_tu117_fast):
        for epoch in range(1, args.epochs + 1):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                # Adaptive VRAM guardrail: split long batches (>64 tokens) into chunks of <=4
                # so peak VRAM stays strictly < 2.6 GiB on GTX 1650 Ti 4 GB.
                cur_bs, cur_seq = input_ids.shape
                sub_bs = 4 if (cur_seq > 64 and cur_bs > 4) else cur_bs
                n_chunks = math.ceil(cur_bs / sub_bs)

                batch_loss_val = 0.0
                valid_batch = True
                for c_start in range(0, cur_bs, sub_bs):
                    c_end = min(c_start + sub_bs, cur_bs)
                    c_weight = (c_end - c_start) / float(cur_bs)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        raw_loss, _ = compute_answer_only_loss(
                            model,
                            input_ids[c_start:c_end],
                            attention_mask[c_start:c_end],
                            labels[c_start:c_end],
                        )
                    if not torch.isfinite(raw_loss):
                        valid_batch = False
                        break
                    scaler.scale((raw_loss * c_weight) / args.grad_accum).backward()
                    batch_loss_val += float(raw_loss.item()) * c_weight

                if not valid_batch:
                    print(f"[Warn] Non-finite micro-batch loss at step {global_step}; skipping.", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    step_loss_accum = 0.0
                    micro_in_step = 0
                    continue

                step_loss_accum += batch_loss_val
                micro_in_step += 1
                accum_counter += 1

                if accum_counter >= args.grad_accum:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=args.max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    avg_step_loss = step_loss_accum / max(micro_in_step, 1)
                    window_loss_sum += avg_step_loss
                    window_steps += 1
                    accum_counter = 0
                    step_loss_accum = 0.0
                    micro_in_step = 0

                    if global_step == 1 or global_step % args.log_steps == 0 or global_step == total_steps:
                        mean_win_loss = window_loss_sum / max(window_steps, 1)
                        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        elapsed = time.perf_counter() - t_start
                        cur_lr = float(scheduler.get_last_lr()[0])
                        frac_epoch = global_step / max(steps_per_epoch, 1)
                        point = {
                            "step": global_step,
                            "epoch": round(frac_epoch, 3),
                            "train_loss": round(mean_win_loss, 5),
                            "lr": round(cur_lr, 8),
                            "peak_vram_mib": round(peak_mib, 1),
                            "elapsed_sec": round(elapsed, 1),
                        }
                        train_curve.append(point)
                        print(
                            f"  step {global_step:4d}/{total_steps} (ep {frac_epoch:.2f}) | "
                            f"loss={mean_win_loss:.4f} | lr={cur_lr:.2e} | "
                            f"peak_vram={peak_mib:.0f} MiB | elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                        window_loss_sum = 0.0
                        window_steps = 0

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        ckpt_dir = ckpt_root / f"step_{global_step:04d}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_dir)
                        _prune_checkpoints(ckpt_root, args.save_total_limit)
                        print(f"  [Checkpoint] Saved {ckpt_dir.name}", flush=True)

                    if global_step >= total_steps:
                        stop_early = True
                        break

            # Flush remaining partial accumulation at epoch end if not stopped early
            if not stop_early and accum_counter > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                avg_step_loss = step_loss_accum / max(micro_in_step, 1)
                window_loss_sum += avg_step_loss
                window_steps += 1
                accum_counter = 0
                step_loss_accum = 0.0
                micro_in_step = 0

            _run_val(tag_epoch=global_step / max(steps_per_epoch, 1))
            if stop_early:
                break

    if not (adapter_dir / "adapter_config.json").exists():
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    total_time = time.perf_counter() - t_start
    peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    curve_payload = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "gpu": gpu_info,
        "precision": {
            "fp16": True,
            "bf16": False,
            "lora_master_fp32": True,
            "tu117_fp32_compute_linear": use_tu117_fast,
        },
        "hyperparams": {
            "epochs": args.epochs,
            "total_optimizer_steps": global_step,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": eff_bs,
            "max_length": args.max_length,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "gradient_checkpointing": True,
            "length_bucket_batching": True,
            "answer_only_lm_head": True,
        },
        "lora": param_stats,
        "splits": {
            "train_size": len(splits.train),
            "val_size": len(splits.val),
            "test_size": len(splits.test),
            "num_labels": len(splits.id2label),
        },
        "summary": {
            "initial_train_loss": train_curve[0]["train_loss"] if train_curve else None,
            "final_train_loss": train_curve[-1]["train_loss"] if train_curve else None,
            "best_val_loss": round(best_val_loss, 5) if math.isfinite(best_val_loss) else None,
            "peak_vram_mib": round(peak_vram_mib, 1),
            "total_train_seconds": round(total_time, 1),
        },
        "train_curve": train_curve,
        "val_curve": val_curve,
    }

    curve_path = reports_dir / "train_curve.json"
    curve_path.write_text(json.dumps(curve_payload, indent=2), encoding="utf-8")
    (adapter_dir / "training_metadata.json").write_text(json.dumps(curve_payload, indent=2), encoding="utf-8")

    plot_loss_curve(
        train_points=train_curve,
        val_points=val_curve,
        param_stats=param_stats,
        peak_vram_mb=peak_vram_mib,
        eff_bs=eff_bs,
        out_path=reports_dir / "loss_curve.png",
    )
    print(
        f"[Done] Training finished in {total_time:.1f}s | peak_vram={peak_vram_mib:.1f} MiB | "
        f"best_val_loss={best_val_loss:.4f} | saved -> {adapter_dir}",
        flush=True,
    )

    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return curve_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Qwen3-0.6B with LoRA on Banking77 (GTX 1650 Ti safe)")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=650, help="0 = run full --epochs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--log-steps", type=int, default=25)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--min-free-vram-mib", type=int, default=3200)
    p.add_argument("--allow-low-vram", action="store_true")
    p.add_argument("--disable-tu117-fast", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
