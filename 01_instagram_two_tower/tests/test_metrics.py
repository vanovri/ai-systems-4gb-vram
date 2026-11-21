"""Unit tests for exact Recall@K, NDCG@K, MRR, Coverage@10, Popularity Gini, and Head-Item Share."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval import (
    compute_gini_coefficient,
    compute_ranking_and_bias_metrics,
    mrr_at_k_single,
    ndcg_at_k_single,
    recall_at_k_single,
)


def test_recall_at_k_exact_when_liked_lt_k() -> None:
    rec = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    liked = {20, 50, 999}  # |liked| = 3 < K = 10 -> denom = min(3, 10) = 3
    assert recall_at_k_single(rec, liked, k=10) == pytest.approx(2.0 / 3.0)
    # At K=2: top_2 = [10, 20] -> 1 hit, denom = min(3, 2) = 2 -> 0.5
    assert recall_at_k_single(rec, liked, k=2) == pytest.approx(0.5)


def test_recall_at_k_exact_when_liked_gt_k() -> None:
    rec = [0, 1, 2, 3, 4]
    liked = {1, 3, 10, 11, 12, 13, 14, 15, 16, 17}  # |liked| = 10 > K = 5 -> denom = min(10, 5) = 5
    assert recall_at_k_single(rec, liked, k=5) == pytest.approx(2.0 / 5.0)
    assert recall_at_k_single(rec, set(), k=5) == 0.0


def test_ndcg_and_mrr_exact_math() -> None:
    # Hit at rank 1 (idx 0) and rank 3 (idx 2); |liked| = 2 -> IDCG@5 = 1/log2(2) + 1/log2(3)
    rec = [5, 9, 7, 1, 2]
    liked = {5, 7}
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k_single(rec, liked, k=5) == pytest.approx(dcg / idcg, rel=1e-6)
    assert mrr_at_k_single(rec, liked, k=5) == pytest.approx(1.0)

    # First hit at rank 4 -> MRR = 1/4 = 0.25
    assert mrr_at_k_single([9, 8, 7, 5, 1], {5}, k=5) == pytest.approx(0.25)
    assert mrr_at_k_single([9, 8, 7], {5}, k=3) == 0.0


def test_gini_coverage_and_head_item_share() -> None:
    # Uniform exposure across 100 items -> Gini == 0.0
    assert compute_gini_coefficient(np.ones(100, dtype=np.int64)) == pytest.approx(0.0, abs=1e-9)
    # All exposure concentrated on 1 item out of 100 -> Gini == (100 - 1) / 100 = 0.99
    concentrated = np.zeros(100, dtype=np.int64)
    concentrated[0] = 500
    assert compute_gini_coefficient(concentrated) == pytest.approx(0.99, abs=1e-6)

    # Full ranking + bias evaluation on synthetic 20-item catalog
    num_items = 20
    # Item 0 is the Top-5% head item (5% of 20 = 1 item)
    item_pos_counts = np.arange(num_items, 0, -1, dtype=np.int64)
    # 2 users, each recommended 10 items; User 0 gets [0..9], User 1 gets [0, 10..18]
    recs = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ],
        dtype=np.int64,
    )
    gt = {0: {0, 9}, 1: {10, 19}}
    m = compute_ranking_and_bias_metrics(
        recommendations=recs,
        eval_user_ids=[0, 1],
        ground_truth_liked=gt,
        item_pos_counts=item_pos_counts,
        num_items=num_items,
        k_values=(10,),
    )
    # 19 unique items recommended out of 20 -> coverage@10 = 0.95
    assert m["coverage@10"] == pytest.approx(19.0 / 20.0)
    # Item 0 appears twice out of 20 recommendation slots -> head_item_share_top5pct@10 = 2/20 = 0.10
    assert m["head_item_share_top5pct@10"] == pytest.approx(0.10)
    # User 0 Recall@10 = 2/2 = 1.0, User 1 Recall@10 = 1/2 = 0.5 -> mean = 0.75
    assert m["recall@10"] == pytest.approx(0.75)
