"""Visualizations for Project #4: Layer-10 INT4 snap levels (64 real weights vs 16 levels) and summary charts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.quantizer import (  # noqa: E402
    compute_quant_metrics,
    dequantize_groupwise,
    get_group_snap_levels,
    quantize_groupwise,
)


def plot_layer10_int4_snap(
    w64: np.ndarray,
    out_path: Path,
    layer_label: str = "Qwen3-0.6B — model.layers.10.self_attn.q_proj.weight[0, :64]",
) -> Dict[str, Any]:
    """Generate the signature slide-#5 chart: 64 real Layer-10 weights vs 16 allowed 4-bit snap lines."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(w64, dtype=np.float32).ravel()[:64]
    if w.size != 64:
        raise ValueError(f"Expected 64 weights for group_size=64 visualization, got {w.size}")

    qt = quantize_groupwise(w, bits=4, group_size=64, mode="asymmetric", pack=True)
    w_hat, levels = get_group_snap_levels(w, bits=4, mode="asymmetric")
    metrics = compute_quant_metrics(w, w_hat, qt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=160)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Left panel: Histogram of 64 real FP16 weights + 16 allowed 4-bit snap lines
    ax1.hist(
        w,
        bins=22,
        color="#38bdf8",
        alpha=0.68,
        edgecolor="#0ea5e9",
        linewidth=0.9,
        label="64 Real Layer-10 Weights (FP16)",
    )
    for idx, lv in enumerate(levels):
        ax1.axvline(
            float(lv),
            color="#f59e0b",
            linestyle="--",
            linewidth=1.05,
            alpha=0.82,
            label="16 Allowed INT4 Snap Levels" if idx == 0 else None,
        )
    # Rug plot along bottom showing exact quantized weight positions
    y_top = ax1.get_ylim()[1]
    ax1.scatter(w_hat, np.full_like(w_hat, y_top * 0.03), color="#34d399", s=16, zorder=5, label="Snapped INT4 Values")
    ax1.set_title("Distribution of 64 Weights vs 16 INT4 Snap Levels", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Weight Value", color="#cbd5e1", fontsize=10)
    ax1.set_ylabel("Count", color="#cbd5e1", fontsize=10)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.22, color="#94a3b8")
    leg1 = ax1.legend(loc="upper right", fontsize=8.3, facecolor="#1e293b", edgecolor="#475569")
    for txt in leg1.get_texts():
        txt.set_color("#f1f5f9")

    # Right panel: Weight index 0..63 FP16 -> Snapped INT4 displacement against the 16 gridlines
    xs = np.arange(64)
    for lv in levels:
        ax2.axhline(float(lv), color="#f59e0b", linestyle="--", linewidth=0.75, alpha=0.35)
    for i in range(64):
        ax2.plot([i, i], [float(w[i]), float(w_hat[i])], color="#94a3b8", linewidth=0.9, alpha=0.65)
    ax2.scatter(xs, w, color="#38bdf8", s=24, zorder=4, label="Original FP16 Weight")
    ax2.scatter(xs, w_hat, color="#34d399", marker="x", s=34, linewidths=1.4, zorder=5, label="Snapped INT4 Weight")

    ax2.set_title("Per-Weight Snapping Error (Group Size = 64)", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_xlabel("Weight Index Inside Group (0..63)", color="#cbd5e1", fontsize=10)
    ax2.set_ylabel("Weight Value", color="#cbd5e1", fontsize=10)
    ax2.set_xlim(-1.5, 64.5)
    leg2 = ax2.legend(loc="upper right", fontsize=8.3, facecolor="#1e293b", edgecolor="#475569")
    for txt in leg2.get_texts():
        txt.set_color("#f1f5f9")

    stats_box = (
        f"scale (FP16) = {float(qt.scales[0]):.6f} | min (FP16) = {float(qt.zeros[0]):.6f}\n"
        f"Rel L1 err ε = {metrics['rel_l1_error']*100:.2f}% | RMSE = {metrics['rmse']:.6f}\n"
        f"Packed size = {qt.total_nbytes} B (32B nibbles + 4B meta) vs {qt.fp16_baseline_nbytes} B FP16 ({qt.compression_ratio_vs_fp16:.2f}x)"
    )
    ax2.text(
        0.02,
        0.03,
        stats_box,
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#e2e8f0",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#475569", alpha=0.92),
    )

    fig.suptitle(
        f"4-Bit Asymmetric Group-Wise Quantization — {layer_label}",
        color="#f8fafc",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        "layer": layer_label,
        "num_weights": 64,
        "num_levels": 16,
        "scale_fp16": float(qt.scales[0]),
        "min_fp16": float(qt.zeros[0]),  # type: ignore[index]
        "rel_l1_error": round(metrics["rel_l1_error"], 6),
        "rmse": round(metrics["rmse"], 6),
        "max_abs_error": round(metrics["max_abs_error"], 6),
        "packed_bytes": qt.total_nbytes,
        "fp16_bytes": qt.fp16_baseline_nbytes,
        "compression_ratio": round(qt.compression_ratio_vs_fp16, 4),
        "plot_path": str(out_path),
    }


def plot_benchmark_summary(bench: Dict[str, Any], out_path: Path) -> None:
    """Create a 3-panel summary chart (WikiText-2 PPL, Quantized Linear Size MB, Weight Rel L1 Error %)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variants = bench["wikitext2_benchmark"]
    labels = ["FP16\n(Baseline)", "INT8 Asym\n(g=64)", "INT4 Asym\n(g=64)", "INT4 Sym\n(g=64)"]
    keys = ["fp16", "int8_asym_g64", "int4_asym_g64", "int4_sym_g64"]
    present = [(lbl, k) for lbl, k in zip(labels, keys) if k in variants]
    if not present:
        return

    lbls = [p[0] for p in present]
    ppls = [variants[p[1]]["perplexity"] for p in present]
    lin_mbs = [variants[p[1]]["quantized_linear_mb"] for p in present]
    l1_pcts = [variants[p[1]]["mean_rel_l1_error_pct"] for p in present]
    colors = ["#38bdf8", "#34d399", "#f59e0b", "#a78bfa"][: len(present)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=160)
    fig.patch.set_facecolor("#0f172a")

    panels = [
        (axes[0], ppls, "WikiText-2 Perplexity (lower is better)", "Perplexity", "{:.2f}"),
        (axes[1], lin_mbs, "Quantized Linear Weight Size (MB)", "Megabytes (packed + meta)", "{:.1f} MB"),
        (axes[2], l1_pcts, "Mean Relative L1 Weight Error ε (%)", "Rel L1 Error (%)", "{:.2f}%"),
    ]
    for ax, vals, title, ylabel, fmt in panels:
        ax.set_facecolor("#111827")
        bars = ax.bar(lbls, vals, color=colors, width=0.58, edgecolor="#e2e8f0", linewidth=0.7)
        ax.set_title(title, color="#f8fafc", fontsize=10.5, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=9.5)
        ax.tick_params(colors="#cbd5e1", labelsize=8.8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.22, color="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        top = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0, top * 1.22)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + top * 0.03,
                fmt.format(val),
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=8.8,
                fontweight="bold",
            )

    fig.suptitle(
        "Qwen3-0.6B Pure-NumPy Group-Wise Quantization Benchmark (group_size=64)",
        color="#f8fafc",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)
    w64 = model.model.layers[10].self_attn.q_proj.weight[0, :64].detach().cpu().float().numpy()
    info = plot_layer10_int4_snap(w64, PROJECT_DIR / "reports" / "layer10_int4_snap.png")
    print(json.dumps(info, indent=2))
