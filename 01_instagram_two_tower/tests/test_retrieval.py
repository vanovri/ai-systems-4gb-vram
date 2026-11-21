"""Unit tests for FastTwoTowerIndex top-K parity and retrieval latency benchmark."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.retrieval import FastTwoTowerIndex


def test_retrieval_index_exact_parity_with_bruteforce() -> None:
    torch.manual_seed(42)
    num_items = 250
    dim = 64
    tau = 0.05
    beta = 0.65

    item_embs = F.normalize(torch.randn(num_items, dim), p=2, dim=-1)
    user_embs = F.normalize(torch.randn(5, dim), p=2, dim=-1)
    probs = torch.linspace(1e-4, 1e-2, num_items)
    probs = probs / probs.sum()
    log_p = torch.log(probs)

    index = FastTwoTowerIndex(
        item_embeddings=item_embs,
        tau=tau,
        log_item_probs=log_p,
        infer_logq_beta=beta,
        device="cpu",
    )

    exclude_sets = [{0, 5, 10}, {3}, set(), {1, 2, 3, 4}, {249}]
    batch_ids, batch_scores = index.retrieve_batch(user_embs, top_k=15, exclude_sets=exclude_sets)

    # Brute-force reference in NumPy
    u_np = user_embs.numpy()
    v_np = item_embs.numpy()
    lp_np = log_p.numpy()

    for row_idx in range(5):
        ref_cos = u_np[row_idx] @ v_np.T
        ref_scores = ref_cos / tau - beta * lp_np
        for ex in exclude_sets[row_idx]:
            ref_scores[ex] = -1e9
        ref_top = np.argsort(-ref_scores)[:15]

        # Compare batch retrieval
        assert np.array_equal(batch_ids[row_idx], ref_top)
        assert np.allclose(batch_scores[row_idx], ref_scores[ref_top], atol=1e-4)

        # Compare single-user retrieval
        s_ids, s_scores, s_cos = index.retrieve_single(
            user_embs[row_idx], top_k=15, exclude_ids=exclude_sets[row_idx]
        )
        assert np.array_equal(s_ids, ref_top)
        assert np.allclose(s_scores, ref_scores[ref_top], atol=1e-4)
        assert np.allclose(s_cos, ref_cos[ref_top], atol=1e-5)
        for ex in exclude_sets[row_idx]:
            assert ex not in set(s_ids.tolist())


def test_retrieval_latency_benchmark_valid_quantiles() -> None:
    torch.manual_seed(0)
    item_embs = F.normalize(torch.randn(500, 64), p=2, dim=-1)
    user_embs = F.normalize(torch.randn(20, 64), p=2, dim=-1)
    index = FastTwoTowerIndex(item_embeddings=item_embs, tau=0.05, device="cpu")
    lat = index.benchmark_latency(user_embs, top_k=10, num_queries=60, warmup=10)

    assert lat["catalog_size"] == 500
    assert lat["top_k"] == 10
    assert lat["p50_ms"] > 0.0
    assert lat["p50_ms"] <= lat["p95_ms"] <= lat["p99_ms"]
    assert lat["qps"] > 0.0
