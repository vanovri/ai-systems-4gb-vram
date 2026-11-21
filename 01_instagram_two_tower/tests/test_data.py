"""Unit tests for data pipeline: chronological leakage-free split, train-only features, Zipfian skew, KuaiRec loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DataConfig
from src.data import (
    chronological_per_user_split,
    extract_train_only_features,
    generate_synthetic_interactions,
    load_or_generate_dataset,
)


def test_chronological_split_zero_temporal_leakage() -> None:
    cfg = DataConfig(num_users=80, num_items=200, num_creators=30, interactions_per_user=40, seed=123)
    bundle = load_or_generate_dataset(cfg)

    train_max = bundle.train_df.groupby("user_id")["timestamp"].max()
    val_min = bundle.val_df.groupby("user_id")["timestamp"].min()
    val_max = bundle.val_df.groupby("user_id")["timestamp"].max()
    test_min = bundle.test_df.groupby("user_id")["timestamp"].min()

    common_tv = train_max.index.intersection(val_min.index)
    assert len(common_tv) == cfg.num_users
    assert (train_max.loc[common_tv] <= val_min.loc[common_tv]).all()

    common_vt = val_max.index.intersection(test_min.index)
    assert len(common_vt) == cfg.num_users
    assert (val_max.loc[common_vt] <= test_min.loc[common_vt]).all()

    # Check train, val, and test items are disjoint per user
    for u in range(cfg.num_users):
        tr_items = bundle.train_seen.get(u, set())
        te_liked = bundle.test_liked.get(u, set())
        assert tr_items.isdisjoint(te_liked)


def test_train_only_feature_extraction_has_no_val_test_leakage() -> None:
    cfg = DataConfig(num_users=60, num_items=150, num_creators=25, interactions_per_user=30, seed=7)
    interactions_df, items_meta_df, topic_names, _, _ = generate_synthetic_interactions(cfg)
    train_df, val_df, test_df = chronological_per_user_split(interactions_df)

    res1 = extract_train_only_features(train_df, items_meta_df, cfg.num_users, cfg.num_items, len(topic_names))

    # Corrupt val_df & test_df completely; train-derived features must remain identical
    val_df["liked"] = 1 - val_df["liked"]
    test_df["watch_ratio"] = 999.0
    res2 = extract_train_only_features(train_df, items_meta_df, cfg.num_users, cfg.num_items, len(topic_names))

    for arr1, arr2 in zip(res1, res2):
        assert np.array_equal(arr1, arr2)


def test_synthetic_zipfian_popularity_skew_and_finite_probs() -> None:
    cfg = DataConfig(num_users=250, num_items=500, num_creators=60, interactions_per_user=50, seed=99)
    bundle = load_or_generate_dataset(cfg)

    assert bundle.stats["total_interactions"] == 250 * 50
    # Top 5% items must have substantially higher share than uniform (0.05)
    assert bundle.stats["train_top5pct_pos_share"] > 0.20
    assert np.isfinite(bundle.log_item_probs).all()
    assert np.isclose(bundle.item_probs.sum(), 1.0, atol=1e-5)
    assert (bundle.item_probs > 0.0).all()


def test_kuairec_loader_parses_real_csv_schema(tmp_path: Path) -> None:
    # Build minimal valid KuaiRec 2.0 CSV fixture on disk
    rng = np.random.default_rng(42)
    rows = []
    ts = 1_700_000_000
    for u in range(100, 120):
        vids = rng.choice(np.arange(1000, 1060), size=20, replace=False)
        for step, vid in enumerate(vids):
            wr = float(rng.uniform(0.1, 1.8))
            rows.append(
                {
                    "user_id": u,
                    "video_id": int(vid),
                    "play_duration": int(wr * 20000),
                    "video_duration": 20000,
                    "timestamp": ts + step * 60,
                    "watch_ratio": wr,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "small_matrix.csv", index=False)

    cat_rows = [{"video_id": int(v), "feat": f"[{int(v % 8)}, {int((v + 1) % 8)}]"} for v in range(1000, 1060)]
    pd.DataFrame(cat_rows).to_csv(tmp_path / "item_categories.csv", index=False)

    feat_rows = [{"video_id": int(v), "author_id": int(500 + (v % 12))} for v in range(1000, 1060)]
    pd.DataFrame(feat_rows).to_csv(tmp_path / "item_daily_features.csv", index=False)

    cfg = DataConfig(dataset="kuairec", data_dir=str(tmp_path))
    bundle = load_or_generate_dataset(cfg)

    assert bundle.dataset_name == "kuairec"
    assert bundle.num_users == 20
    assert bundle.num_items == 60
    assert bundle.num_topics == 8
    assert bundle.num_creators == 12
    assert bundle.stats["total_interactions"] == 400
    assert np.isfinite(bundle.user_dense).all()
    assert np.isfinite(bundle.item_dense).all()
