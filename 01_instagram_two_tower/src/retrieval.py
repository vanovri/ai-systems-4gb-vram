"""Fast Top-K Retrieval Index over Precomputed Item Tower Embeddings.

Supports exact cosine top-K search, optional serving-time Log-Q popularity debiasing,
seen-item masking, batched retrieval, and p50/p95/p99 single-query latency benchmarking.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch

from .model import TwoTowerModel


class FastTwoTowerIndex:
    """In-memory exact vector index backed by precomputed unit-norm ItemTower embeddings."""

    def __init__(
        self,
        item_embeddings: torch.Tensor,
        tau: float = 0.05,
        log_item_probs: torch.Tensor | None = None,
        infer_logq_beta: float = 0.0,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else item_embeddings.device
        self.item_embeddings = item_embeddings.detach().to(self.device, dtype=torch.float32).contiguous()
        self.num_items, self.embed_dim = self.item_embeddings.shape
        self.tau = float(tau)
        self.infer_logq_beta = float(infer_logq_beta)
        if log_item_probs is not None:
            self.log_item_probs = log_item_probs.detach().to(self.device, dtype=torch.float32).contiguous()
            # Precompute additive serving bias vector: -beta * log(p_item)
            self._serving_bias = (-self.infer_logq_beta * self.log_item_probs) if self.infer_logq_beta != 0.0 else None
        else:
            self.log_item_probs = None
            self._serving_bias = None

    @classmethod
    @torch.no_grad()
    def from_model(
        cls,
        model: TwoTowerModel,
        item_topic_ids: np.ndarray,
        item_creator_ids: np.ndarray,
        item_dense: np.ndarray,
        log_item_probs: np.ndarray | None = None,
        infer_logq_beta: float = 0.0,
        batch_size: int = 1024,
        device: torch.device | str | None = None,
    ) -> "FastTwoTowerIndex":
        """Precompute the full catalog item embedding matrix `[num_items, embed_dim]`."""
        model_device = next(model.parameters()).device
        target_device = torch.device(device) if device is not None else model_device
        was_training = model.training
        model.eval()

        num_items = len(item_topic_ids)
        chunks: list[torch.Tensor] = []
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            i_ids = torch.arange(start, end, dtype=torch.long, device=model_device)
            t_ids = torch.as_tensor(item_topic_ids[start:end], dtype=torch.long, device=model_device)
            c_ids = torch.as_tensor(item_creator_ids[start:end], dtype=torch.long, device=model_device)
            dense = torch.as_tensor(item_dense[start:end], dtype=torch.float32, device=model_device)
            v = model.encode_items(i_ids, t_ids, c_ids, dense)
            chunks.append(v.detach().to(target_device, dtype=torch.float32))

        if was_training:
            model.train()

        all_v = torch.cat(chunks, dim=0).contiguous()
        lp_tensor = (
            torch.as_tensor(log_item_probs, dtype=torch.float32, device=target_device)
            if log_item_probs is not None
            else None
        )
        return cls(
            item_embeddings=all_v,
            tau=model.tau,
            log_item_probs=lp_tensor,
            infer_logq_beta=infer_logq_beta,
            device=target_device,
        )

    @torch.no_grad()
    def score_all(self, user_embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(ranking_scores, raw_cosine_scores)` of shape `[B, num_items]`."""
        u = user_embs.to(self.device, dtype=torch.float32)
        if u.ndim == 1:
            u = u.unsqueeze(0)
        cosine = u @ self.item_embeddings.T  # [B, N]
        ranking = cosine / self.tau
        if self._serving_bias is not None:
            ranking = ranking + self._serving_bias.unsqueeze(0)
        return ranking, cosine

    @torch.no_grad()
    def retrieve_single(
        self,
        user_emb: torch.Tensor,
        top_k: int = 10,
        exclude_ids: Iterable[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve top-K item IDs, ranking scores, and raw cosine similarities for one user."""
        ranking, cosine = self.score_all(user_emb)
        ranking_1d = ranking.squeeze(0).clone()
        cosine_1d = cosine.squeeze(0)

        if exclude_ids is not None:
            excl_list = [int(i) for i in exclude_ids if 0 <= int(i) < self.num_items]
            if excl_list:
                ranking_1d[excl_list] = -1e9

        k = min(int(top_k), self.num_items)
        top_scores, top_indices = torch.topk(ranking_1d, k=k, largest=True, sorted=True)
        top_cosines = cosine_1d[top_indices]
        return (
            top_indices.cpu().numpy().astype(np.int64),
            top_scores.cpu().numpy().astype(np.float32),
            top_cosines.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def retrieve_batch(
        self,
        user_embs: torch.Tensor,
        top_k: int = 50,
        exclude_sets: list[set[int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-K item IDs and ranking scores for a batch of users `[B, D]`."""
        ranking, _ = self.score_all(user_embs)
        if exclude_sets is not None:
            ranking = ranking.clone()
            for row_idx, excl in enumerate(exclude_sets):
                if excl:
                    valid = [int(i) for i in excl if 0 <= int(i) < self.num_items]
                    if valid:
                        ranking[row_idx, valid] = -1e9
        k = min(int(top_k), self.num_items)
        top_scores, top_indices = torch.topk(ranking, k=k, dim=1, largest=True, sorted=True)
        return (
            top_indices.cpu().numpy().astype(np.int64),
            top_scores.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def benchmark_latency(
        self,
        user_embs: torch.Tensor,
        top_k: int = 10,
        num_queries: int = 500,
        warmup: int = 50,
    ) -> dict[str, float]:
        """Measure single-query Top-K retrieval latency (p50, p95, p99, mean ms & QPS)."""
        u_all = user_embs.to(self.device, dtype=torch.float32).contiguous()
        n_avail = u_all.size(0)
        k = min(int(top_k), self.num_items)
        is_cuda = self.device.type == "cuda"

        # Warmup
        for i in range(warmup):
            u = u_all[i % n_avail].unsqueeze(0)
            s = (u @ self.item_embeddings.T) / self.tau
            if self._serving_bias is not None:
                s = s + self._serving_bias.unsqueeze(0)
            _ = torch.topk(s, k=k, dim=1)
        if is_cuda:
            torch.cuda.synchronize(self.device)

        latencies_ms = np.empty(num_queries, dtype=np.float64)
        for i in range(num_queries):
            u = u_all[i % n_avail].unsqueeze(0)
            if is_cuda:
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            s = (u @ self.item_embeddings.T) / self.tau
            if self._serving_bias is not None:
                s = s + self._serving_bias.unsqueeze(0)
            _ = torch.topk(s, k=k, dim=1)
            if is_cuda:
                torch.cuda.synchronize(self.device)
            latencies_ms[i] = (time.perf_counter() - t0) * 1000.0

        mean_ms = float(latencies_ms.mean())
        return {
            "device": str(self.device),
            "catalog_size": int(self.num_items),
            "top_k": int(k),
            "num_queries": int(num_queries),
            "p50_ms": float(np.percentile(latencies_ms, 50)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
            "mean_ms": mean_ms,
            "qps": float(1000.0 / max(mean_ms, 1e-9)),
        }
