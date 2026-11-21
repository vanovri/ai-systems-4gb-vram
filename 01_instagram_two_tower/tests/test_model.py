"""Unit tests for UserTower, ItemTower, TwoTowerModel, temperature scaling, and Log-Q correction."""

from __future__ import annotations

import pytest
import torch

from src.config import ModelConfig
from src.model import TwoTowerModel, apply_serving_logq_adjustment


def _make_model() -> TwoTowerModel:
    cfg = ModelConfig(embed_dim=64, hidden_dim=128, proj_hidden_dim=64, dropout=0.1, temperature=0.05)
    return TwoTowerModel(
        num_users=50,
        num_items=100,
        num_topics=12,
        num_creators=20,
        user_dense_dim=17,
        item_dense_dim=6,
        cfg=cfg,
    )


def test_l2_norm_both_towers() -> None:
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    u = model.encode_users(torch.arange(16), torch.randn(16, 17))
    v = model.encode_items(
        torch.arange(16),
        torch.randint(0, 12, (16,)),
        torch.randint(0, 20, (16,)),
        torch.randn(16, 6),
    )
    assert u.shape == (16, 64)
    assert v.shape == (16, 64)
    u_norms = torch.linalg.norm(u, dim=-1)
    v_norms = torch.linalg.norm(v, dim=-1)
    assert torch.allclose(u_norms, torch.ones_like(u_norms), atol=1e-5)
    assert torch.allclose(v_norms, torch.ones_like(v_norms), atol=1e-5)


@pytest.mark.parametrize("variant", ["baseline", "debiased"])
def test_gradient_flow_both_towers(variant: str) -> None:
    torch.manual_seed(7)
    model = _make_model()
    model.train()

    u_ids = torch.arange(16)
    i_ids = torch.arange(16)
    u_dense = torch.randn(16, 17)
    t_ids = torch.randint(0, 12, (16,))
    c_ids = torch.randint(0, 20, (16,))
    i_dense = torch.randn(16, 6)

    u = model.encode_users(u_ids, u_dense)
    v = model.encode_items(i_ids, t_ids, c_ids, i_dense)

    if variant == "baseline":
        loss = model.compute_baseline_loss(u, v, item_ids=i_ids)
    else:
        log_p = torch.linspace(-2.0, -7.0, 16)
        loss = model.compute_debiased_loss(u, v, log_item_probs=log_p, item_ids=i_ids, alpha=1.0)

    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


def test_temperature_scaled_cosine_scoring() -> None:
    torch.manual_seed(1)
    model = _make_model()
    u = model.encode_users(torch.arange(8), torch.randn(8, 17))
    v = model.encode_items(
        torch.arange(8),
        torch.randint(0, 12, (8,)),
        torch.randint(0, 20, (8,)),
        torch.randn(8, 6),
    )
    scores = model.score_matrix(u, v)
    expected = (u @ v.T) / 0.05
    assert torch.allclose(scores, expected, atol=1e-5)
    assert scores.max().item() <= 20.0 + 1e-4
    assert scores.min().item() >= -20.0 - 1e-4


def test_logq_serving_adjustment_suppresses_viral_head_items() -> None:
    # Equal cosine similarity (0.50) across 3 items with head, mid, and tail probabilities
    cos = torch.tensor([[0.50, 0.50, 0.50]], dtype=torch.float32)
    probs = torch.tensor([0.10, 0.01, 0.0005], dtype=torch.float32)
    log_p = torch.log(probs)

    raw_scores = apply_serving_logq_adjustment(cos, tau=0.05, log_item_probs=log_p, beta=0.0)
    deb_scores = apply_serving_logq_adjustment(cos, tau=0.05, log_item_probs=log_p, beta=0.7)

    # Without debiasing, all scores are tied at 0.50 / 0.05 = 10.0
    assert torch.allclose(raw_scores, torch.full_like(raw_scores, 10.0), atol=1e-5)
    # With beta > 0, tail item (index 2) outranks mid (index 1) which outranks viral head (index 0)
    assert deb_scores[0, 2] > deb_scores[0, 1] > deb_scores[0, 0]


def test_debiased_loss_with_mixed_extra_negatives_and_duplicates() -> None:
    torch.manual_seed(3)
    model = _make_model()
    # Intentionally include duplicate item IDs inside batch & overlapping extra negative IDs
    u_ids = torch.arange(8)
    i_ids = torch.tensor([0, 0, 1, 2, 3, 3, 4, 5], dtype=torch.long)
    u = model.encode_users(u_ids, torch.randn(8, 17))
    v = model.encode_items(i_ids, torch.zeros(8, dtype=torch.long), torch.zeros(8, dtype=torch.long), torch.randn(8, 6))
    extra_ids = torch.tensor([0, 10, 11, 12], dtype=torch.long)
    extra_v = model.encode_items(
        extra_ids,
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.randn(4, 6),
    )

    log_p = torch.full((8,), -4.0)
    extra_lp = torch.full((4,), -6.0)

    loss_base = model.compute_baseline_loss(u, v, item_ids=i_ids)
    loss_deb = model.compute_debiased_loss(
        u,
        v,
        log_item_probs=log_p,
        item_ids=i_ids,
        alpha=1.0,
        extra_v=extra_v,
        extra_item_ids=extra_ids,
        extra_log_probs=extra_lp,
    )
    assert torch.isfinite(loss_base) and loss_base.item() > 0.0
    assert torch.isfinite(loss_deb) and loss_deb.item() > 0.0
