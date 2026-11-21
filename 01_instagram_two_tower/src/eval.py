"""Exact Retrieval & Popularity-Bias Evaluation Suite.

Implements:
  - Recall@K (exact formula: |top_K ∩ liked| / min(|liked|, K)) for K in {10, 20, 50}
  - NDCG@K for K in {10, 20, 50}
  - MRR@10 and MRR@50
  - Tail-Item Recall@10 (recall specifically on non-head liked items)
  - Catalog Coverage@10 (fraction of full catalog appearing in at least one user's Top-10)
  - Popularity Gini@10 (exact Gini coefficient of Top-10 recommendation exposure across catalog)
  - Head-Item Share@10 (share of Top-10 slots occupied by Top 1%, Top 5%, Top 10% most popular train items)
  - Long-Tail Share@10 (share of Top-10 slots occupied by bottom 80% catalog items)
  - Exposure distribution across 10 popularity deciles (for comparison plots)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data import DatasetBundle
from .model import TwoTowerModel
from .retrieval import FastTwoTowerIndex


def recall_at_k_single(recommended: np.ndarray | list[int], liked: set[int], k: int) -> float:
    """Exact Recall@K = |top_K ∩ liked| / min(|liked|, K)."""
    if not liked or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if int(item) in liked)
    denom = min(len(liked), k)
    return float(hits) / float(denom)


def ndcg_at_k_single(recommended: np.ndarray | list[int], liked: set[int], k: int) -> float:
    """Exact binary NDCG@K with log2(rank + 1) discount."""
    if not liked or k <= 0:
        return 0.0
    top_k = recommended[:k]
    dcg = 0.0
    for rank_idx, item in enumerate(top_k):
        if int(item) in liked:
            dcg += 1.0 / np.log2(rank_idx + 2.0)
    ideal_hits = min(len(liked), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2, dtype=np.float64))))
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def mrr_at_k_single(recommended: np.ndarray | list[int], liked: set[int], k: int) -> float:
    """Reciprocal rank of the first liked item within top-K (0.0 if absent)."""
    if not liked or k <= 0:
        return 0.0
    for rank_idx, item in enumerate(recommended[:k]):
        if int(item) in liked:
            return 1.0 / float(rank_idx + 1)
    return 0.0


def compute_gini_coefficient(counts: np.ndarray) -> float:
    """Exact Gini coefficient over non-negative exposure counts across all catalog items.

    G = sum_{i=1..N} (2*i - N - 1) * x_(i) / (N * sum(x))
    Returns 0.0 for uniform exposure, approaching 1.0 for extreme concentration.
    """
    x = np.asarray(counts, dtype=np.float64)
    if x.size == 0:
        return 0.0
    total = float(x.sum())
    if total <= 0.0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = np.sum((2.0 * idx - n - 1.0) * x_sorted) / (n * total)
    return float(np.clip(gini, 0.0, 1.0))


def compute_ranking_and_bias_metrics(
    recommendations: np.ndarray,
    eval_user_ids: list[int],
    ground_truth_liked: dict[int, set[int]],
    item_pos_counts: np.ndarray,
    num_items: int,
    k_values: tuple[int, ...] = (10, 20, 50),
) -> dict[str, Any]:
    """Compute exact ranking, coverage, and popularity-bias metrics from top-K recommendation matrix.

    Args:
        recommendations: int64 array of shape `[num_eval_users, max_k]`.
        eval_user_ids: list of user IDs aligned with rows of `recommendations`.
        ground_truth_liked: mapping user_id -> set of liked item IDs in eval split.
        item_pos_counts: train positive count per item `[num_items]` defining head/tail tiers.
        num_items: total catalog size.
        k_values: K cutoffs to evaluate (e.g. (10, 20, 50)).
    """
    num_eval = len(eval_user_ids)
    assert recommendations.shape[0] == num_eval, "recommendations row count mismatch"

    # Sort items by training popularity descending to define exact head/tail sets & deciles
    pop_order = np.argsort(-item_pos_counts, kind="mergesort")
    pop_rank = np.empty(num_items, dtype=np.int64)
    pop_rank[pop_order] = np.arange(num_items)  # 0 = most popular

    top1_cut = max(1, int(round(0.01 * num_items)))
    top5_cut = max(1, int(round(0.05 * num_items)))
    top10_cut = max(1, int(round(0.10 * num_items)))
    top20_cut = max(1, int(round(0.20 * num_items)))

    head_top5_set = set(pop_order[:top5_cut].tolist())

    metrics: dict[str, Any] = {"num_eval_users": num_eval}
    for k in k_values:
        recalls = []
        ndcgs = []
        for row_idx, u in enumerate(eval_user_ids):
            liked = ground_truth_liked[u]
            rec = recommendations[row_idx]
            recalls.append(recall_at_k_single(rec, liked, k))
            ndcgs.append(ndcg_at_k_single(rec, liked, k))
        metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0

    # MRR@10 and MRR@max_k
    max_k = int(recommendations.shape[1])
    mrr10 = [mrr_at_k_single(recommendations[i], ground_truth_liked[u], min(10, max_k)) for i, u in enumerate(eval_user_ids)]
    mrr_max = [mrr_at_k_single(recommendations[i], ground_truth_liked[u], max_k) for i, u in enumerate(eval_user_ids)]
    metrics["mrr@10"] = float(np.mean(mrr10)) if mrr10 else 0.0
    metrics[f"mrr@{max_k}"] = float(np.mean(mrr_max)) if mrr_max else 0.0

    # Tail-Item Recall@10 (liked items outside Top-5% head)
    tail_recalls = []
    for row_idx, u in enumerate(eval_user_ids):
        tail_liked = ground_truth_liked[u] - head_top5_set
        if tail_liked:
            tail_recalls.append(recall_at_k_single(recommendations[row_idx], tail_liked, min(10, max_k)))
    metrics["tail_recall@10"] = float(np.mean(tail_recalls)) if tail_recalls else 0.0

    # Popularity bias & catalog diversity at Top-10
    top10_flat = recommendations[:, : min(10, max_k)].reshape(-1)
    exposure_counts = np.bincount(top10_flat, minlength=num_items)
    total_slots = max(int(top10_flat.size), 1)

    metrics["coverage@10"] = float(np.count_nonzero(exposure_counts) / float(num_items))
    metrics["unique_items_recommended@10"] = int(np.count_nonzero(exposure_counts))
    metrics["popularity_gini@10"] = compute_gini_coefficient(exposure_counts)

    rec_ranks = pop_rank[top10_flat]
    metrics["head_item_share_top1pct@10"] = float(np.mean(rec_ranks < top1_cut))
    metrics["head_item_share_top5pct@10"] = float(np.mean(rec_ranks < top5_cut))
    metrics["head_item_share_top10pct@10"] = float(np.mean(rec_ranks < top10_cut))
    metrics["long_tail_share_bottom80pct@10"] = float(np.mean(rec_ranks >= top20_cut))
    metrics["avg_recommended_pop_rank@10"] = float(np.mean(rec_ranks + 1))

    # Decile exposure shares (Decile 0 = Top 10% most popular -> Decile 9 = Bottom 10% tail)
    deciles = np.minimum((rec_ranks * 10) // max(num_items, 1), 9)
    decile_counts = np.bincount(deciles, minlength=10)
    metrics["pop_decile_exposure@10"] = (decile_counts / float(total_slots)).tolist()

    # Also compute ground-truth test liked decile shares for reference plotting
    gt_items: list[int] = []
    for u in eval_user_ids:
        gt_items.extend(ground_truth_liked[u])
    if gt_items:
        gt_ranks = pop_rank[np.asarray(gt_items, dtype=np.int64)]
        gt_deciles = np.minimum((gt_ranks * 10) // max(num_items, 1), 9)
        gt_decile_counts = np.bincount(gt_deciles, minlength=10)
        metrics["ground_truth_pop_decile_share"] = (gt_decile_counts / float(len(gt_items))).tolist()
    else:
        metrics["ground_truth_pop_decile_share"] = [0.0] * 10

    return metrics


@torch.no_grad()
def encode_all_users(
    model: TwoTowerModel,
    user_dense: np.ndarray,
    batch_size: int = 512,
) -> torch.Tensor:
    """Compute unit-norm embeddings for all users `[num_users, embed_dim]`."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    num_users = user_dense.shape[0]
    chunks: list[torch.Tensor] = []
    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        u_ids = torch.arange(start, end, dtype=torch.long, device=device)
        dense = torch.as_tensor(user_dense[start:end], dtype=torch.float32, device=device)
        chunks.append(model.encode_users(u_ids, dense).detach().float())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0).contiguous()


@torch.no_grad()
def evaluate_model(
    model: TwoTowerModel,
    bundle: DatasetBundle,
    split: str = "test",
    infer_logq_beta: float = 0.0,
    k_values: tuple[int, ...] = (10, 20, 50),
    batch_size: int = 256,
    run_latency_benchmark: bool = False,
    latency_queries: int = 500,
    latency_warmup: int = 50,
) -> tuple[dict[str, Any], FastTwoTowerIndex]:
    """Full catalog evaluation of TwoTowerModel on `val` or `test` split."""
    if split == "val":
        gt_liked = bundle.val_liked
        seen_map = bundle.train_seen
    elif split == "test":
        gt_liked = bundle.test_liked
        seen_map = bundle.train_val_seen
    else:
        raise ValueError(f"Unknown split '{split}', expected 'val' or 'test'.")

    eval_user_ids = sorted(u for u, liked in gt_liked.items() if len(liked) > 0)
    max_k = max(k_values)

    index = FastTwoTowerIndex.from_model(
        model=model,
        item_topic_ids=bundle.item_topic_ids,
        item_creator_ids=bundle.item_creator_ids,
        item_dense=bundle.item_dense,
        log_item_probs=bundle.log_item_probs,
        infer_logq_beta=infer_logq_beta,
    )

    all_user_embs = encode_all_users(model, bundle.user_dense, batch_size=max(batch_size, 512))

    rec_rows: list[np.ndarray] = []
    for start in range(0, len(eval_user_ids), batch_size):
        batch_u = eval_user_ids[start : start + batch_size]
        u_embs = all_user_embs[batch_u]
        excl = [seen_map.get(u, set()) for u in batch_u]
        top_ids, _ = index.retrieve_batch(u_embs, top_k=max_k, exclude_sets=excl)
        rec_rows.append(top_ids)

    recommendations = np.vstack(rec_rows) if rec_rows else np.zeros((0, max_k), dtype=np.int64)
    metrics = compute_ranking_and_bias_metrics(
        recommendations=recommendations,
        eval_user_ids=eval_user_ids,
        ground_truth_liked=gt_liked,
        item_pos_counts=bundle.item_pos_counts,
        num_items=bundle.num_items,
        k_values=k_values,
    )
    metrics["split"] = split
    metrics["infer_logq_beta"] = float(infer_logq_beta)

    if run_latency_benchmark and len(eval_user_ids) > 0:
        eval_u_embs = all_user_embs[eval_user_ids]
        metrics["latency_gpu_or_active"] = index.benchmark_latency(
            user_embs=eval_u_embs,
            top_k=10,
            num_queries=latency_queries,
            warmup=latency_warmup,
        )
        # Also measure pure CPU retrieval latency for deployment reference
        cpu_index = FastTwoTowerIndex(
            item_embeddings=index.item_embeddings.cpu(),
            tau=index.tau,
            log_item_probs=index.log_item_probs.cpu() if index.log_item_probs is not None else None,
            infer_logq_beta=infer_logq_beta,
            device="cpu",
        )
        metrics["latency_cpu"] = cpu_index.benchmark_latency(
            user_embs=eval_u_embs.cpu(),
            top_k=10,
            num_queries=latency_queries,
            warmup=latency_warmup,
        )

    return metrics, index
