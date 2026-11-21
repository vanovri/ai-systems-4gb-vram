"""Training & Head-to-Head Benchmark CLI for Two-Tower Instagram Reels RecSys.

Trains `baseline` (in-batch softmax without popularity correction) and `debiased`
(Log-Q popularity correction + hard/uniform catalog negative mixing + serving-time Log-Q),
saves best checkpoints to `artifacts/`, writes `reports/benchmark.json` and
`reports/comparison_plot.png`, and frees GPU VRAM cleanly on exit.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainConfig
from .data import DatasetBundle, PositiveInteractionDataset, load_or_generate_dataset
from .eval import evaluate_model
from .model import TwoTowerModel
from .plotting import generate_comparison_plot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def _mine_mixed_hard_negatives(
    model: TwoTowerModel,
    u_batch: torch.Tensor,
    all_item_topic_t: torch.Tensor,
    all_item_creator_t: torch.Tensor,
    all_item_dense_t: torch.Tensor,
    num_items: int,
    extra_neg_count: int,
    hard_neg_pool_factor: int,
    hard_neg_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select mixed catalog negatives (semi-hard negatives + uniform negatives).

    Returns:
        selected_ids: `[extra_neg_count]` int64 item IDs
        selected_log_probs: `[extra_neg_count]` float32 uniform catalog log proposal (-log(N))
    """
    n_hard = min(extra_neg_count, max(0, int(round(extra_neg_count * hard_neg_fraction))))
    n_unif = extra_neg_count - n_hard
    pool_size = min(num_items, max(extra_neg_count, n_hard * max(1, hard_neg_pool_factor)))

    selected_parts: list[torch.Tensor] = []
    if n_hard > 0:
        pool_ids = torch.randint(0, num_items, (pool_size,), device=device, dtype=torch.long)
        was_training = model.training
        model.eval()
        pool_v = model.encode_items(
            pool_ids,
            all_item_topic_t[pool_ids],
            all_item_creator_t[pool_ids],
            all_item_dense_t[pool_ids],
        )
        if was_training:
            model.train()
        # Rank pool items by max/top similarity to batch users (semi-hard negatives)
        sim = u_batch.detach().float() @ pool_v.float().T  # [B, pool_size]
        hardness = sim.topk(k=min(8, sim.size(0)), dim=0).values.mean(dim=0)
        hard_idx = torch.topk(hardness, k=min(n_hard, pool_size), largest=True).indices
        selected_parts.append(pool_ids[hard_idx])

    if n_unif > 0:
        unif_ids = torch.randint(0, num_items, (n_unif,), device=device, dtype=torch.long)
        selected_parts.append(unif_ids)

    selected_ids = torch.cat(selected_parts, dim=0)
    unif_log_p = torch.full((selected_ids.size(0),), -math.log(float(num_items)), dtype=torch.float32, device=device)
    return selected_ids, unif_log_p


def train_variant(
    variant: str,
    bundle: DatasetBundle,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    artifacts_dir: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one Two-Tower variant (`baseline` or `debiased`), checkpoint best weights, and evaluate."""
    if variant not in ("baseline", "debiased"):
        raise ValueError(f"Unknown variant '{variant}'. Expected 'baseline' or 'debiased'.")

    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    use_amp = bool(train_cfg.use_amp_fp16 and device.type == "cuda")

    model = TwoTowerModel(
        num_users=bundle.num_users,
        num_items=bundle.num_items,
        num_topics=bundle.num_topics,
        num_creators=bundle.num_creators,
        user_dense_dim=bundle.user_dense.shape[1],
        item_dense_dim=bundle.item_dense.shape[1],
        cfg=model_cfg,
    ).to(device)

    # Pre-stage static feature lookup tables on active device (compact: < 2 MB total)
    user_dense_t = torch.as_tensor(bundle.user_dense, dtype=torch.float32, device=device)
    item_dense_t = torch.as_tensor(bundle.item_dense, dtype=torch.float32, device=device)
    item_topic_t = torch.as_tensor(bundle.item_topic_ids, dtype=torch.long, device=device)
    item_creator_t = torch.as_tensor(bundle.item_creator_ids, dtype=torch.long, device=device)
    log_item_probs_t = torch.as_tensor(bundle.log_item_probs, dtype=torch.float32, device=device)

    dataset = PositiveInteractionDataset(bundle.train_pos_users, bundle.train_pos_items)
    gen = torch.Generator()
    gen.manual_seed(train_cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=len(dataset) > train_cfg.batch_size,
        generator=gen,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(train_cfg.epochs, 1), eta_min=train_cfg.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    infer_beta = float(train_cfg.infer_logq_beta) if variant == "debiased" else 0.0
    extra_neg_count = int(train_cfg.extra_neg_count) if variant == "debiased" else 0

    best_score = -1.0
    best_epoch = 0
    best_val_metrics: dict[str, Any] = {}
    ckpt_path = artifacts_dir / f"{variant}_best.pt"
    history: list[dict[str, Any]] = []

    t_start = time.perf_counter()
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        num_steps = 0

        for u_ids, i_ids in loader:
            u_ids = u_ids.to(device, non_blocking=True)
            i_ids = i_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                u = model.encode_users(u_ids, user_dense_t[u_ids])
                v = model.encode_items(i_ids, item_topic_t[i_ids], item_creator_t[i_ids], item_dense_t[i_ids])

                if variant == "baseline":
                    loss = model.compute_baseline_loss(u, v, item_ids=i_ids)
                else:
                    extra_v = None
                    extra_ids = None
                    extra_lp = None
                    batch_lp = log_item_probs_t[i_ids]

                    if extra_neg_count > 0:
                        extra_ids, unif_lp = _mine_mixed_hard_negatives(
                            model=model,
                            u_batch=u,
                            all_item_topic_t=item_topic_t,
                            all_item_creator_t=item_creator_t,
                            all_item_dense_t=item_dense_t,
                            num_items=bundle.num_items,
                            extra_neg_count=extra_neg_count,
                            hard_neg_pool_factor=train_cfg.hard_neg_pool_factor,
                            hard_neg_fraction=train_cfg.hard_neg_fraction,
                            device=device,
                        )
                        extra_v = model.encode_items(
                            extra_ids,
                            item_topic_t[extra_ids],
                            item_creator_t[extra_ids],
                            item_dense_t[extra_ids],
                        )
                        # Exact mixture proposal log-weights between B in-batch negatives and M catalog negatives
                        b_cnt = float(i_ids.size(0))
                        m_cnt = float(extra_ids.size(0))
                        batch_lp = batch_lp + math.log(b_cnt / (b_cnt + m_cnt))
                        extra_lp = unif_lp + math.log(m_cnt / (b_cnt + m_cnt))

                    loss = model.compute_debiased_loss(
                        u=u,
                        v=v,
                        log_item_probs=batch_lp,
                        item_ids=i_ids,
                        alpha=train_cfg.logq_alpha,
                        extra_v=extra_v,
                        extra_item_ids=extra_ids,
                        extra_log_probs=extra_lp,
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().item())
            num_steps += 1

        scheduler.step()
        avg_loss = total_loss / max(num_steps, 1)
        row: dict[str, Any] = {"epoch": epoch, "train_loss": avg_loss, "lr": float(scheduler.get_last_lr()[0])}

        if epoch % train_cfg.eval_every == 0 or epoch == train_cfg.epochs:
            val_metrics, _ = evaluate_model(
                model=model,
                bundle=bundle,
                split="val",
                infer_logq_beta=infer_beta,
                k_values=eval_cfg.k_values,
                batch_size=eval_cfg.batch_size,
                run_latency_benchmark=False,
            )
            row["val_recall@10"] = val_metrics["recall@10"]
            row["val_ndcg@10"] = val_metrics["ndcg@10"]
            row["val_coverage@10"] = val_metrics["coverage@10"]
            row["val_gini@10"] = val_metrics["popularity_gini@10"]

            score = val_metrics["ndcg@10"] + 0.5 * val_metrics["recall@10"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_val_metrics = val_metrics
                torch.save(
                    {
                        "variant": variant,
                        "epoch": epoch,
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_config": model_cfg.__dict__,
                        "infer_logq_beta": infer_beta,
                        "logq_alpha": float(train_cfg.logq_alpha) if variant == "debiased" else 0.0,
                        "num_users": bundle.num_users,
                        "num_items": bundle.num_items,
                        "num_topics": bundle.num_topics,
                        "num_creators": bundle.num_creators,
                        "user_dense_dim": int(bundle.user_dense.shape[1]),
                        "item_dense_dim": int(bundle.item_dense.shape[1]),
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
            if verbose:
                print(
                    f"[{variant:8s}] epoch {epoch:02d}/{train_cfg.epochs:02d} | "
                    f"loss={avg_loss:.4f} | val R@10={val_metrics['recall@10']:.4f} | "
                    f"NDCG@10={val_metrics['ndcg@10']:.4f} | Cov@10={val_metrics['coverage@10']:.4f} | "
                    f"Gini={val_metrics['popularity_gini@10']:.4f}"
                )
        history.append(row)

    train_seconds = time.perf_counter() - t_start

    # Reload best checkpoint for final unbiased test evaluation + latency benchmark
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics, _ = evaluate_model(
        model=model,
        bundle=bundle,
        split="test",
        infer_logq_beta=infer_beta,
        k_values=eval_cfg.k_values,
        batch_size=eval_cfg.batch_size,
        run_latency_benchmark=True,
        latency_queries=eval_cfg.latency_queries,
        latency_warmup=eval_cfg.latency_warmup,
    )

    extra_evals: dict[str, Any] = {}
    if variant == "debiased":
        # Also evaluate pure training-time Log-Q (beta=0.0, Yi et al. 2019) and a Pareto sweep over serving beta
        logq_train_only_metrics, _ = evaluate_model(
            model=model,
            bundle=bundle,
            split="test",
            infer_logq_beta=0.0,
            k_values=eval_cfg.k_values,
            batch_size=eval_cfg.batch_size,
            run_latency_benchmark=False,
        )
        extra_evals["logq_train_only_test_metrics"] = logq_train_only_metrics

        beta_grid = sorted({0.0, 0.15, 0.25, float(round(infer_beta, 2)), 0.50, 0.65})
        pareto_sweep: list[dict[str, Any]] = []
        for b in beta_grid:
            if abs(b - 0.0) < 1e-6:
                bm_pt = logq_train_only_metrics
            elif abs(b - infer_beta) < 1e-6:
                bm_pt = test_metrics
            else:
                bm_pt, _ = evaluate_model(
                    model=model,
                    bundle=bundle,
                    split="test",
                    infer_logq_beta=b,
                    k_values=eval_cfg.k_values,
                    batch_size=eval_cfg.batch_size,
                    run_latency_benchmark=False,
                )
            pareto_sweep.append(
                {
                    "infer_logq_beta": b,
                    "recall@10": bm_pt["recall@10"],
                    "recall@20": bm_pt["recall@20"],
                    "recall@50": bm_pt["recall@50"],
                    "ndcg@10": bm_pt["ndcg@10"],
                    "tail_recall@10": bm_pt["tail_recall@10"],
                    "coverage@10": bm_pt["coverage@10"],
                    "popularity_gini@10": bm_pt["popularity_gini@10"],
                    "head_item_share_top5pct@10": bm_pt["head_item_share_top5pct@10"],
                    "long_tail_share_bottom80pct@10": bm_pt["long_tail_share_bottom80pct@10"],
                }
            )
        extra_evals["serving_beta_pareto_sweep"] = pareto_sweep

    result = {
        "variant": variant,
        "best_epoch": best_epoch,
        "train_seconds": float(train_seconds),
        "checkpoint_path": str(ckpt_path),
        "logq_alpha": float(train_cfg.logq_alpha) if variant == "debiased" else 0.0,
        "infer_logq_beta": infer_beta,
        "extra_neg_count": extra_neg_count,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        **extra_evals,
    }

    # Explicitly free GPU memory
    del model, optimizer, scheduler, scaler, user_dense_t, item_dense_t, item_topic_t, item_creator_t, log_item_probs_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_serving_metadata(bundle: DatasetBundle, data_cfg: DataConfig, artifacts_dir: Path) -> Path:
    """Save compact train-derived features & metadata so `src/demo.py` loads instantaneously."""
    meta_path = artifacts_dir / "serving_bundle.pt"
    torch.save(
        {
            "dataset_name": bundle.dataset_name,
            "data_config": data_cfg.__dict__,
            "num_users": bundle.num_users,
            "num_items": bundle.num_items,
            "num_topics": bundle.num_topics,
            "num_creators": bundle.num_creators,
            "topic_names": bundle.topic_names,
            "user_dense": bundle.user_dense,
            "item_dense": bundle.item_dense,
            "item_topic_ids": bundle.item_topic_ids,
            "item_creator_ids": bundle.item_creator_ids,
            "item_durations": bundle.item_durations,
            "item_train_counts": bundle.item_train_counts,
            "item_pos_counts": bundle.item_pos_counts,
            "item_probs": bundle.item_probs,
            "log_item_probs": bundle.log_item_probs,
            "train_seen": bundle.train_seen,
            "item_titles": bundle.item_titles,
            "creator_names": bundle.creator_names,
            "user_favorite_creators": bundle.user_favorite_creators,
            "stats": bundle.stats,
        },
        meta_path,
    )
    return meta_path


def run_benchmark(
    exp_cfg: ExperimentConfig,
    variants: tuple[str, ...] = ("baseline", "debiased"),
    project_root: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run complete data prep, head-to-head training, evaluation, JSON report, and plotting."""
    root = project_root if project_root is not None else Path(__file__).resolve().parent.parent
    artifacts_dir, reports_dir = exp_cfg.resolve_paths(root)

    bundle = load_or_generate_dataset(exp_cfg.data)
    meta_path = save_serving_metadata(bundle, exp_cfg.data, artifacts_dir)

    if verbose:
        print(
            f"Loaded {bundle.dataset_name}: {bundle.num_users:,} users x {bundle.num_items:,} videos, "
            f"{bundle.stats['total_interactions']:,} interactions "
            f"(train pos={bundle.stats['train_positive_pairs']:,}, "
            f"top-5% head pos share={bundle.stats['train_top5pct_pos_share']:.1%})"
        )

    device = resolve_device(exp_cfg.train.device)
    hardware_info = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "amp_fp16": bool(exp_cfg.train.use_amp_fp16 and device.type == "cuda"),
        "torch_version": torch.__version__,
    }

    variant_results: dict[str, Any] = {}
    for variant in variants:
        variant_results[variant] = train_variant(
            variant=variant,
            bundle=bundle,
            model_cfg=exp_cfg.model,
            train_cfg=exp_cfg.train,
            eval_cfg=exp_cfg.eval,
            artifacts_dir=artifacts_dir,
            verbose=verbose,
        )

    deltas: dict[str, float] = {}
    deltas_vs_logq0: dict[str, float] = {}
    if "baseline" in variant_results and "debiased" in variant_results:
        bm = variant_results["baseline"]["test_metrics"]
        dm = variant_results["debiased"]["test_metrics"]
        l0 = variant_results["debiased"].get("logq_train_only_test_metrics", {})
        for key in (
            "recall@10",
            "recall@20",
            "recall@50",
            "ndcg@10",
            "mrr@10",
            "tail_recall@10",
            "coverage@10",
            "popularity_gini@10",
            "head_item_share_top5pct@10",
            "long_tail_share_bottom80pct@10",
        ):
            if key in bm and key in dm:
                deltas[f"delta_{key}"] = float(dm[key] - bm[key])
            if key in l0 and key in dm:
                deltas_vs_logq0[f"delta_{key}"] = float(dm[key] - l0[key])

    benchmark = {
        "project": "01_instagram_two_tower",
        "hardware": hardware_info,
        "dataset_stats": bundle.stats,
        "data_config": exp_cfg.data.__dict__,
        "model_config": exp_cfg.model.__dict__,
        "train_config": exp_cfg.train.__dict__,
        "serving_bundle_path": str(meta_path),
        "variants": variant_results,
        "deltas_debiased_minus_baseline": deltas,
        "deltas_debiased_minus_logq_train_only": deltas_vs_logq0,
    }

    bench_path = reports_dir / "benchmark.json"
    bench_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

    plot_path: Path | None = None
    if "baseline" in variant_results and "debiased" in variant_results:
        plot_path = generate_comparison_plot(benchmark, reports_dir / "comparison_plot.png")
        benchmark["comparison_plot_path"] = str(plot_path)
        bench_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"Saved benchmark report: {bench_path}")
        if plot_path is not None:
            print(f"Saved comparison plot:  {plot_path}")

    return benchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train & benchmark Two-Tower Instagram Reels RecSys.")
    p.add_argument("--dataset", choices=["synthetic", "kuairec"], default="synthetic")
    p.add_argument("--data-dir", type=str, default=None, help="Path to KuaiRec CSV folder (for --dataset kuairec)")
    p.add_argument("--variant", choices=["baseline", "debiased", "both"], default="both")
    p.add_argument("--num-users", type=int, default=2000)
    p.add_argument("--num-items", type=int, default=5000)
    p.add_argument("--interactions-per-user", type=int, default=115)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--logq-alpha", type=float, default=1.0)
    p.add_argument("--infer-logq-beta", type=float, default=0.35)
    p.add_argument("--extra-neg-count", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable FP16 AMP even on CUDA")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = ExperimentConfig(
        data=DataConfig(
            dataset=args.dataset,
            data_dir=args.data_dir,
            num_users=args.num_users,
            num_items=args.num_items,
            interactions_per_user=args.interactions_per_user,
            seed=args.seed,
        ),
        model=ModelConfig(temperature=args.tau),
        train=TrainConfig(
            variant=args.variant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            logq_alpha=args.logq_alpha,
            infer_logq_beta=args.infer_logq_beta,
            extra_neg_count=args.extra_neg_count,
            use_amp_fp16=not args.no_amp,
            seed=args.seed,
            device=args.device,
        ),
        eval=EvalConfig(),
    )
    variants = ("baseline", "debiased") if args.variant == "both" else (args.variant,)
    run_benchmark(exp_cfg, variants=variants, verbose=True)


if __name__ == "__main__":
    main()
