"""Interactive Demo & Programmatic `predict(user_id, top_k=10)` CLI for Two-Tower Instagram Reels RecSys."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .model import TwoTowerModel
from .retrieval import FastTwoTowerIndex


def _popularity_tier(item_id: int, pop_rank: np.ndarray, num_items: int) -> str:
    pct = (int(pop_rank[item_id]) + 1) / float(max(num_items, 1))
    if pct <= 0.05:
        return "Head (Top 5%)"
    if pct <= 0.20:
        return "Mid-Tail (5-20%)"
    return "Long-Tail (Bottom 80%)"


def _build_match_reason(
    user_id: int,
    item_id: int,
    bundle: dict[str, Any],
    pop_rank: np.ndarray,
    is_debiased: bool = True,
) -> str:
    topic_id = int(bundle["item_topic_ids"][item_id])
    creator_id = int(bundle["item_creator_ids"][item_id])
    topic_name = bundle["topic_names"][topic_id]
    creator_name = bundle["creator_names"][creator_id]
    # User topic preference vector occupies the last `num_topics` columns of user_dense
    num_topics = int(bundle["num_topics"])
    u_topic_vec = bundle["user_dense"][user_id, -num_topics:]
    topic_share = float(u_topic_vec[topic_id])

    reasons: list[str] = []
    if topic_share >= 0.15:
        reasons.append(f"strong topic affinity ({topic_name}: {topic_share:.0%} history)")
    elif topic_share >= 0.06:
        reasons.append(f"topic match ({topic_name}: {topic_share:.0%} history)")
    else:
        reasons.append(f"cross-topic discovery ({topic_name})")

    fav_creators = bundle.get("user_favorite_creators", {}).get(user_id, set())
    if creator_id in fav_creators:
        reasons.append(f"favorite creator {creator_name}")

    # Compare user preferred normalized log duration (index 4 of user_dense) vs item normalized log duration (index 0)
    u_dur = float(bundle["user_dense"][user_id, 4])
    i_dur = float(bundle["item_dense"][item_id, 0])
    if abs(u_dur - i_dur) < 0.65:
        reasons.append(f"duration fit ({float(bundle['item_durations'][item_id]):.0f}s)")

    tier = _popularity_tier(item_id, pop_rank, int(bundle["num_items"]))
    if tier.startswith("Long-Tail"):
        reasons.append("long-tail gem surfaced via Log-Q" if is_debiased else "long-tail catalog item")
    return " + ".join(reasons)


def predict(
    user_id: int,
    top_k: int = 10,
    variant: str = "debiased",
    artifacts_dir: Path | str | None = None,
    device: str = "cpu",
    exclude_seen: bool = True,
) -> list[dict[str, Any]]:
    """Generate Top-K video recommendations with cosine similarity scores & human-readable match reasons."""
    root = Path(__file__).resolve().parent.parent
    art_dir = Path(artifacts_dir) if artifacts_dir is not None else (root / "artifacts")
    ckpt_path = art_dir / f"{variant}_best.pt"
    bundle_path = art_dir / "serving_bundle.pt"

    if not ckpt_path.exists() or not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing trained artifacts ({ckpt_path} or {bundle_path}). "
            "Run `python -m src.train --variant both` first."
        )

    dev = torch.device(device)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    num_users = int(bundle["num_users"])
    if not (0 <= user_id < num_users):
        raise ValueError(f"user_id={user_id} out of range [0, {num_users - 1}]")

    model_cfg = ModelConfig(**ckpt["model_config"])
    model = TwoTowerModel(
        num_users=ckpt["num_users"],
        num_items=ckpt["num_items"],
        num_topics=ckpt["num_topics"],
        num_creators=ckpt["num_creators"],
        user_dense_dim=ckpt["user_dense_dim"],
        item_dense_dim=ckpt["item_dense_dim"],
        cfg=model_cfg,
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    index = FastTwoTowerIndex.from_model(
        model=model,
        item_topic_ids=bundle["item_topic_ids"],
        item_creator_ids=bundle["item_creator_ids"],
        item_dense=bundle["item_dense"],
        log_item_probs=bundle["log_item_probs"],
        infer_logq_beta=float(ckpt.get("infer_logq_beta", 0.0)),
        device=dev,
    )

    with torch.no_grad():
        u_id_t = torch.tensor([user_id], dtype=torch.long, device=dev)
        u_dense_t = torch.as_tensor(bundle["user_dense"][user_id : user_id + 1], dtype=torch.float32, device=dev)
        u_emb = model.encode_users(u_id_t, u_dense_t)

    excl = bundle["train_seen"].get(user_id, set()) if exclude_seen else None
    top_ids, top_scores, top_cosines = index.retrieve_single(u_emb, top_k=top_k, exclude_ids=excl)

    item_pos_counts = np.asarray(bundle["item_pos_counts"], dtype=np.int64)
    pop_order = np.argsort(-item_pos_counts, kind="mergesort")
    pop_rank = np.empty(int(bundle["num_items"]), dtype=np.int64)
    pop_rank[pop_order] = np.arange(int(bundle["num_items"]))

    recs: list[dict[str, Any]] = []
    for rank, (iid, score, cos) in enumerate(zip(top_ids, top_scores, top_cosines), start=1):
        iid_int = int(iid)
        topic_id = int(bundle["item_topic_ids"][iid_int])
        creator_id = int(bundle["item_creator_ids"][iid_int])
        recs.append(
            {
                "rank": rank,
                "item_id": iid_int,
                "title": bundle["item_titles"][iid_int],
                "topic": bundle["topic_names"][topic_id],
                "creator": bundle["creator_names"][creator_id],
                "duration_sec": round(float(bundle["item_durations"][iid_int]), 1),
                "cosine_score": round(float(cos), 4),
                "ranking_score": round(float(score), 4),
                "train_likes": int(item_pos_counts[iid_int]),
                "popularity_tier": _popularity_tier(iid_int, pop_rank, int(bundle["num_items"])),
                "match_reason": _build_match_reason(
                    user_id, iid_int, bundle, pop_rank, is_debiased=(float(ckpt.get("infer_logq_beta", 0.0)) > 0.0)
                ),
            }
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return recs


def _print_recs(user_id: int, variant: str, recs: list[dict[str, Any]]) -> None:
    print(f"\n=== Top-{len(recs)} Recommendations for User #{user_id} [{variant.upper()}] ===")
    for r in recs:
        print(
            f"  #{r['rank']:02d} | vid={r['item_id']:4d} | cos={r['cosine_score']:+.4f} | "
            f"score={r['ranking_score']:6.2f} | {r['popularity_tier']:22s} | "
            f"{r['title']} ({r['creator']}, {r['duration_sec']:.0f}s)\n"
            f"       Reason: {r['match_reason']}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Demo CLI for Two-Tower Instagram Reels Recommendations.")
    p.add_argument("--user-id", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--variant", choices=["baseline", "debiased", "compare"], default="compare")
    p.add_argument("--artifacts-dir", type=str, default=None)
    args = p.parse_args()

    if args.variant == "compare":
        for v in ("baseline", "debiased"):
            recs = predict(user_id=args.user_id, top_k=args.top_k, variant=v, artifacts_dir=args.artifacts_dir)
            _print_recs(args.user_id, v, recs)
    else:
        recs = predict(user_id=args.user_id, top_k=args.top_k, variant=args.variant, artifacts_dir=args.artifacts_dir)
        _print_recs(args.user_id, args.variant, recs)


if __name__ == "__main__":
    main()
