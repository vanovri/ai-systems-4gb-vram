"""Publication-grade comparison visualization for Baseline vs Log-Q Two-Tower RecSys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_comparison_plot(benchmark: dict[str, Any], output_path: Path | str) -> Path:
    """Create a 2x2 diagnostic figure comparing Baseline, Log-Q Train-Only (Yi 2019), and Full Debiased."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    base = benchmark["variants"]["baseline"]["test_metrics"]
    deb_entry = benchmark["variants"]["debiased"]
    deb = deb_entry["test_metrics"]
    logq0 = deb_entry.get("logq_train_only_test_metrics", deb)
    sweep = deb_entry.get("serving_beta_pareto_sweep", [])
    beta_val = deb_entry.get("infer_logq_beta", 0.35)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.alpha": 0.8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")

    c_base = "#E11D48"  # Rose-red: uncorrected in-batch softmax baseline
    c_logq0 = "#F59E0B"  # Amber: Log-Q training correction only (beta=0, Yi et al. 2019)
    c_deb = "#0EA5E9"  # Sky-blue: Full Two-Stage Log-Q Debiased (train alpha + serving beta)
    c_gt = "#334155"  # Slate: ground truth test likes

    # ---- Panel A: Ranking Accuracy (Recall@K, NDCG@10, Tail Recall@10) ----
    ax = axes[0, 0]
    acc_labels = ["Recall@10", "Recall@20", "Recall@50", "NDCG@10", "Tail Rec@10"]
    acc_keys = ["recall@10", "recall@20", "recall@50", "ndcg@10", "tail_recall@10"]
    b_vals = [base.get(k, 0.0) for k in acc_keys]
    l0_vals = [logq0.get(k, 0.0) for k in acc_keys]
    d_vals = [deb.get(k, 0.0) for k in acc_keys]

    x = np.arange(len(acc_labels))
    w = 0.26
    bars_b = ax.bar(x - w, b_vals, width=w, label="Baseline (In-Batch Softmax)", color=c_base, alpha=0.88)
    bars_l = ax.bar(x, l0_vals, width=w, label="Log-Q Train Only (β=0.0)", color=c_logq0, alpha=0.90)
    bars_d = ax.bar(x + w, d_vals, width=w, label=f"Log-Q Debiased (β={beta_val:.2f})", color=c_deb, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(acc_labels, fontsize=9.5)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("A. Retrieval Accuracy (Test Split)", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_ylim(0.0, max(max(b_vals + l0_vals + d_vals) * 1.24, 0.10))
    ax.grid(axis="y")
    ax.legend(frameon=True, facecolor="#F8FAFC", fontsize=8.4)
    for bar in list(bars_b) + list(bars_l) + list(bars_d):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=7.3)

    # ---- Panel B: Popularity Bias & Catalog Diversity ----
    ax = axes[0, 1]
    bias_labels = ["Coverage@10\n(higher=better)", "Pop Gini@10\n(lower=better)", "Head Top-5%\nShare (lower)", "Long-Tail 80%\nShare (higher)"]
    bias_keys = [
        "coverage@10",
        "popularity_gini@10",
        "head_item_share_top5pct@10",
        "long_tail_share_bottom80pct@10",
    ]
    b_bias = [base.get(k, 0.0) for k in bias_keys]
    l0_bias = [logq0.get(k, 0.0) for k in bias_keys]
    d_bias = [deb.get(k, 0.0) for k in bias_keys]

    x2 = np.arange(len(bias_labels))
    bars_b2 = ax.bar(x2 - w, b_bias, width=w, label="Baseline (In-Batch)", color=c_base, alpha=0.88)
    bars_l2 = ax.bar(x2, l0_bias, width=w, label="Log-Q Train Only (β=0.0)", color=c_logq0, alpha=0.90)
    bars_d2 = ax.bar(x2 + w, d_bias, width=w, label=f"Log-Q Debiased (β={beta_val:.2f})", color=c_deb, alpha=0.92)
    ax.set_xticks(x2)
    ax.set_xticklabels(bias_labels, fontsize=9.0)
    ax.set_ylabel("Ratio / Coefficient [0, 1]", fontsize=10)
    ax.set_title("B. Popularity Bias Mitigation & Catalog Coverage@10", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_ylim(0.0, 1.15)
    ax.grid(axis="y")
    ax.legend(frameon=True, facecolor="#F8FAFC", fontsize=8.4)
    for bar in list(bars_b2) + list(bars_l2) + list(bars_d2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.014, f"{h:.3f}", ha="center", va="bottom", fontsize=7.4)

    # ---- Panel C: Exposure Distribution Across Popularity Deciles ----
    ax = axes[1, 0]
    deciles = np.arange(1, 11)
    b_dec = np.asarray(base.get("pop_decile_exposure@10", [0.0] * 10), dtype=np.float64) * 100.0
    l0_dec = np.asarray(logq0.get("pop_decile_exposure@10", [0.0] * 10), dtype=np.float64) * 100.0
    d_dec = np.asarray(deb.get("pop_decile_exposure@10", [0.0] * 10), dtype=np.float64) * 100.0
    gt_dec = np.asarray(base.get("ground_truth_pop_decile_share", [0.0] * 10), dtype=np.float64) * 100.0

    ax.plot(deciles, b_dec, marker="o", linewidth=2.0, color=c_base, label="Baseline (In-Batch)")
    ax.plot(deciles, l0_dec, marker="D", linewidth=2.0, color=c_logq0, label="Log-Q Train Only (β=0.0)")
    ax.plot(deciles, d_dec, marker="s", linewidth=2.3, color=c_deb, label=f"Log-Q Debiased (β={beta_val:.2f})")
    ax.plot(deciles, gt_dec, marker="^", linewidth=1.8, linestyle="--", color=c_gt, label="Test Ground-Truth Likes")
    ax.set_xticks(deciles)
    ax.set_xticklabels([f"D{i}" for i in deciles], fontsize=9.0)
    ax.set_xlabel("Item Popularity Decile (D1 = Top 10% Viral Head  →  D10 = Deep Long Tail)", fontsize=9.5)
    ax.set_ylabel("Share of Top-10 Recommendations (%)", fontsize=10)
    ax.set_title("C. Recommendation Exposure by Item Popularity Decile", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="#F8FAFC", fontsize=8.4)

    # ---- Panel D: Accuracy vs Popularity Bias Pareto Frontier (Serving Log-Q beta sweep) ----
    ax = axes[1, 1]
    if sweep:
        betas = [pt["infer_logq_beta"] for pt in sweep]
        r10s = [pt["recall@10"] for pt in sweep]
        ginis = [pt["popularity_gini@10"] for pt in sweep]
        covs = [pt["coverage@10"] for pt in sweep]
        heads = [pt["head_item_share_top5pct@10"] for pt in sweep]

        l1 = ax.plot(betas, r10s, marker="s", linewidth=2.2, color=c_deb, label="Recall@10")
        ax.axhline(base.get("recall@10", 0.0), color=c_base, linestyle="--", linewidth=1.4, label="Baseline Recall@10")
        ax.set_xlabel("Serving Log-Q Debiasing Strength (β)", fontsize=10)
        ax.set_ylabel("Recall@10", fontsize=10, color=c_deb)
        ax.tick_params(axis="y", labelcolor=c_deb)
        ax.grid(True)

        ax2 = ax.twinx()
        l2 = ax2.plot(betas, ginis, marker="o", linewidth=2.0, color="#9333EA", label="Popularity Gini@10")
        l3 = ax2.plot(betas, heads, marker="^", linewidth=2.0, color="#EA580C", label="Head Top-5% Share")
        l4 = ax2.plot(betas, covs, marker="D", linewidth=1.8, linestyle=":", color="#16A34A", label="Coverage@10")
        ax2.set_ylabel("Bias / Coverage [0, 1]", fontsize=10)
        ax2.set_ylim(0.0, 1.05)

        ax.axvline(beta_val, color="#0F172A", linestyle="-.", linewidth=1.1, alpha=0.6)
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines] + ["Baseline Recall@10"]
        handles = lines + [ax.lines[-2]]
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=True,
            facecolor="#F8FAFC",
            fontsize=7.6,
        )
    ax.set_title("D. Recall@10 vs Popularity Bias Pareto Frontier (Serving β)", fontsize=11.5, fontweight="bold", pad=40)

    ds_stats = benchmark.get("dataset_stats", {})
    hw = benchmark.get("hardware", {})
    subtitle = (
        f"Dataset: {ds_stats.get('dataset', 'synthetic')} | "
        f"{ds_stats.get('num_users', 0):,} users × {ds_stats.get('num_items', 0):,} videos "
        f"({ds_stats.get('total_interactions', 0):,} interactions) | "
        f"Device: {hw.get('device_name', 'CPU')} | tau={benchmark.get('model_config', {}).get('temperature', 0.05)}"
    )
    fig.suptitle(
        "Two-Tower Instagram Reels Retrieval: In-Batch Sampling Bias & Log-Q Popularity Debiasing",
        fontsize=13.8,
        fontweight="bold",
        y=0.985,
    )
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=9.8, color="#475569")

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.925], h_pad=2.4)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
