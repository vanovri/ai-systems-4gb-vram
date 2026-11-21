"""Two-Tower Neural Retrieval Model with Log-Q Popularity Debiasing.

Architecture:
  UserTower: [user_id_emb, user_dense] -> Linear+LayerNorm+ReLU -> Dropout -> Linear+ReLU -> Linear(64) -> L2-norm
  ItemTower: [item_id_emb, topic_emb, creator_emb, item_dense] -> Linear+LayerNorm+ReLU -> Dropout -> Linear+ReLU -> Linear(64) -> L2-norm
  Scoring:   scores = (u @ v.T) / tau   (tau = 0.05)

Loss variants:
  - baseline: In-batch softmax F.cross_entropy(scores, labels) without popularity correction.
  - debiased: In-batch softmax with Log-Q sampling bias correction
              (logits = scores - alpha * log(p_item)) + optional mixed/hard negative sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class UserTower(nn.Module):
    """Encodes user ID + train-derived behavioral profile into a unit-norm 64-d vector."""

    def __init__(
        self,
        num_users: int,
        user_dense_dim: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.user_id_emb = nn.Embedding(num_users, cfg.user_id_emb_dim)
        in_dim = cfg.user_id_emb_dim + user_dense_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.proj_hidden_dim, cfg.embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_id_emb.weight, mean=0.0, std=0.05)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, user_dense: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_id_emb(user_ids)
        x = torch.cat([u_emb, user_dense], dim=-1)
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1, eps=1e-8)


class ItemTower(nn.Module):
    """Encodes video ID + topic + creator + duration/engagement stats into a unit-norm 64-d vector."""

    def __init__(
        self,
        num_items: int,
        num_topics: int,
        num_creators: int,
        item_dense_dim: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.item_id_emb = nn.Embedding(num_items, cfg.item_id_emb_dim)
        self.topic_emb = nn.Embedding(num_topics, cfg.topic_emb_dim)
        self.creator_emb = nn.Embedding(num_creators, cfg.creator_emb_dim)
        in_dim = cfg.item_id_emb_dim + cfg.topic_emb_dim + cfg.creator_emb_dim + item_dense_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.proj_hidden_dim, cfg.embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_id_emb.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.topic_emb.weight, mean=0.0, std=0.08)
        nn.init.normal_(self.creator_emb.weight, mean=0.0, std=0.06)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        item_ids: torch.Tensor,
        topic_ids: torch.Tensor,
        creator_ids: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        i_emb = self.item_id_emb(item_ids)
        t_emb = self.topic_emb(topic_ids)
        c_emb = self.creator_emb(creator_ids)
        x = torch.cat([i_emb, t_emb, c_emb, item_dense], dim=-1)
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1, eps=1e-8)


class TwoTowerModel(nn.Module):
    """Dual-encoder Two-Tower model with temperature-scaled cosine similarity and Log-Q correction."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_topics: int,
        num_creators: int,
        user_dense_dim: int,
        item_dense_dim: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tau = float(cfg.temperature)
        self.user_tower = UserTower(num_users, user_dense_dim, cfg)
        self.item_tower = ItemTower(num_items, num_topics, num_creators, item_dense_dim, cfg)

    def encode_users(self, user_ids: torch.Tensor, user_dense: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_ids, user_dense)

    def encode_items(
        self,
        item_ids: torch.Tensor,
        topic_ids: torch.Tensor,
        creator_ids: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(item_ids, topic_ids, creator_ids, item_dense)

    def score_matrix(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute temperature-scaled cosine similarity matrix `(u @ v.T) / tau` in FP32."""
        return (u.float() @ v.float().T) / self.tau

    def compute_baseline_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard in-batch softmax loss WITHOUT popularity correction."""
        scores = self.score_matrix(u, v)
        batch_size = scores.size(0)
        if item_ids is not None:
            # Mask off-diagonal accidental duplicate items (false negatives) while preserving diagonal positive
            same_item = item_ids.unsqueeze(1) == item_ids.unsqueeze(0)
            eye = torch.eye(batch_size, dtype=torch.bool, device=scores.device)
            false_neg_mask = same_item & (~eye)
            scores = scores.masked_fill(false_neg_mask, -1e4)
        labels = torch.arange(batch_size, device=scores.device)
        return F.cross_entropy(scores, labels)

    def compute_debiased_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        log_item_probs: torch.Tensor,
        item_ids: torch.Tensor | None = None,
        alpha: float = 1.0,
        extra_v: torch.Tensor | None = None,
        extra_item_ids: torch.Tensor | None = None,
        extra_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """In-batch softmax loss WITH Log-Q correction (`logits = scores - alpha * log(p_item)`)
        and optional mixed/hard catalog negatives.
        """
        scores = self.score_matrix(u, v)
        batch_size = scores.size(0)
        log_p = log_item_probs.float().unsqueeze(0)  # [1, B]
        logits = scores - float(alpha) * log_p

        if item_ids is not None:
            same_item = item_ids.unsqueeze(1) == item_ids.unsqueeze(0)
            eye = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(same_item & (~eye), -1e4)

        if extra_v is not None and extra_v.size(0) > 0:
            extra_scores = self.score_matrix(u, extra_v)  # [B, M]
            if extra_log_probs is not None:
                extra_logits = extra_scores - float(alpha) * extra_log_probs.float().unsqueeze(0)
            else:
                extra_logits = extra_scores
            if item_ids is not None and extra_item_ids is not None:
                collision = item_ids.unsqueeze(1) == extra_item_ids.unsqueeze(0)
                extra_logits = extra_logits.masked_fill(collision, -1e4)
            logits = torch.cat([logits, extra_logits], dim=1)

        labels = torch.arange(batch_size, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        user_ids: torch.Tensor,
        user_dense: torch.Tensor,
        item_ids: torch.Tensor,
        topic_ids: torch.Tensor,
        creator_ids: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.encode_users(user_ids, user_dense)
        v = self.encode_items(item_ids, topic_ids, creator_ids, item_dense)
        scores = self.score_matrix(u, v)
        return u, v, scores


def apply_serving_logq_adjustment(
    cosine_scores: torch.Tensor,
    tau: float,
    log_item_probs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Apply serving-time Log-Q popularity debiasing: `scores = (u @ v.T)/tau - beta * log(p_item)`.

    Note: Since p_item in (0, 1), log(p_item) < 0.
    For head items (larger p_item), log(p_item) is less negative -> smaller (-beta * log p).
    For tail items (smaller p_item), log(p_item) is more negative -> larger (-beta * log p).
    Thus beta > 0 directly counteracts viral head dominance at retrieval time.
    """
    scaled = cosine_scores.float() / float(tau)
    if beta == 0.0:
        return scaled
    return scaled - float(beta) * log_item_probs.float().unsqueeze(0)
