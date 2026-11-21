"""Data pipeline for Two-Tower Instagram Reels RecSys.

Supports:
1. Synthetic realistic short-video benchmark with Zipfian/power-law popularity skew,
   latent topic clusters, creator affinity, duration preferences, and watch/like/skip dynamics.
2. KuaiRec real short-video dataset loader (`--dataset kuairec --data-dir ...`).
3. Strictly leakage-free per-user chronological train/val/test splitting.
4. Train-only user & item feature extraction (zero val/test leakage).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig, TOPICS


TITLE_ADJECTIVES: tuple[str, ...] = (
    "Insane",
    "60-Second",
    "Hidden",
    "Pro",
    "Daily",
    "Ultimate",
    "Minimal",
    "Viral",
    "Beginner",
    "Deep-Dive",
    "Behind-the-Scenes",
    "Weekend",
)

TITLE_NOUNS: tuple[str, ...] = (
    "Breakdown",
    "Routine",
    "Challenge",
    "Tutorial",
    "Hack",
    "Setup",
    "Highlight",
    "Storytime",
    "Transformation",
    "Guide",
    "Experiment",
    "Showcase",
)


@dataclass
class DatasetBundle:
    """Container holding leakage-free splits, features, metadata, and ground truth sets."""

    dataset_name: str
    num_users: int
    num_items: int
    num_topics: int
    num_creators: int
    topic_names: list[str]
    # DataFrames (columns: user_id, item_id, timestamp, watch_ratio, liked, skipped)
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    # Positive-only training pairs (user_id, item_id)
    train_pos_users: np.ndarray
    train_pos_items: np.ndarray
    # Train-only extracted feature tensors/arrays
    user_dense: np.ndarray  # [num_users, user_dense_dim], float32
    item_dense: np.ndarray  # [num_items, item_dense_dim], float32
    item_topic_ids: np.ndarray  # [num_items], int64
    item_creator_ids: np.ndarray  # [num_items], int64
    item_durations: np.ndarray  # [num_items], float32 (seconds)
    # Train-only empirical item popularity distribution
    item_train_counts: np.ndarray  # [num_items], int64 (positive + impression counts)
    item_pos_counts: np.ndarray  # [num_items], int64 (positive interaction counts in train)
    item_probs: np.ndarray  # [num_items], float32 (smoothed positive sampling probability p_item)
    log_item_probs: np.ndarray  # [num_items], float32 (log(p_item))
    # Evaluation mappings (per-user ground truth liked sets & seen-item exclusion masks)
    val_liked: dict[int, set[int]]
    test_liked: dict[int, set[int]]
    train_seen: dict[int, set[int]]
    train_val_seen: dict[int, set[int]]
    # Human-readable item metadata for demo CLI
    item_titles: list[str]
    creator_names: list[str]
    user_favorite_creators: dict[int, set[int]]
    stats: dict[str, Any]


class PositiveInteractionDataset(Dataset):
    """PyTorch Dataset yielding positive (user_id, item_id) training pairs."""

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray) -> None:
        self.user_ids = torch.as_tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.as_tensor(item_ids, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate_synthetic_interactions(
    cfg: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[int, set[int]]]:
    """Generate a realistic Zipfian short-video interaction dataset.

    Returns:
        interactions_df: columns [user_id, item_id, timestamp, watch_ratio, liked, skipped]
        items_meta_df: columns [item_id, topic_id, creator_id, duration_sec, title, quality, pop_rank]
        topic_names: list of topic names
        creator_names: list of creator handles
        user_fav_creators: mapping user_id -> set of favorite creator_ids (for demo explanations)
    """
    rng = np.random.default_rng(cfg.seed)
    num_users = cfg.num_users
    num_items = cfg.num_items
    num_creators = min(cfg.num_creators, num_items)
    num_topics = cfg.num_topics
    topic_names = list(TOPICS[:num_topics])
    if len(topic_names) < num_topics:
        topic_names.extend([f"Topic_{i}" for i in range(len(topic_names), num_topics)])

    # 1. Creators: each specializes in a primary topic + secondary topic
    creator_primary_topic = rng.integers(0, num_topics, size=num_creators)
    creator_secondary_topic = (creator_primary_topic + rng.integers(1, num_topics, size=num_creators)) % num_topics
    creator_names = [f"@{topic_names[creator_primary_topic[c]].lower().split()[0]}_{c:03d}" for c in range(num_creators)]
    # Creator fame follows a mild power-law
    creator_ranks = rng.permutation(num_creators) + 1
    creator_fame = (creator_ranks + 3.0) ** (-0.75)
    creator_fame /= creator_fame.sum()

    # 2. Videos / Items: assign creators, topics, durations, intrinsic quality, and Zipfian viral skew
    item_creator_ids = rng.choice(num_creators, size=num_items, p=creator_fame)
    use_secondary = rng.random(num_items) < 0.18
    item_topic_ids = np.where(
        use_secondary,
        creator_secondary_topic[item_creator_ids],
        creator_primary_topic[item_creator_ids],
    )
    item_durations = np.clip(np.exp(rng.normal(np.log(24.0), 0.62, size=num_items)), 6.0, 120.0).astype(np.float32)
    item_log_dur = np.log(item_durations)
    item_quality = rng.normal(0.0, 1.0, size=num_items).astype(np.float32)

    # Zipfian viral exposure propensity across catalog (decoupled from topic so tail holds high-quality niche gems)
    pop_perm = rng.permutation(num_items)
    item_pop_rank = np.empty(num_items, dtype=np.int64)
    item_pop_rank[pop_perm] = np.arange(1, num_items + 1)
    # Zipf-Mandelbrot power law (small shift q=2.0 gives realistic steep Reels viral head across catalog sizes)
    zipf_weights = (item_pop_rank + 2.0) ** (-cfg.zipf_exponent)
    zipf_weights = (zipf_weights / zipf_weights.sum()).astype(np.float64)

    # Human-readable titles for demo inspection
    adj_idx = rng.integers(0, len(TITLE_ADJECTIVES), size=num_items)
    noun_idx = rng.integers(0, len(TITLE_NOUNS), size=num_items)
    item_titles = [
        f"{TITLE_ADJECTIVES[adj_idx[i]]} {topic_names[item_topic_ids[i]]} {TITLE_NOUNS[noun_idx[i]]} #{i}"
        for i in range(num_items)
    ]

    items_meta_df = pd.DataFrame(
        {
            "item_id": np.arange(num_items, dtype=np.int64),
            "topic_id": item_topic_ids.astype(np.int64),
            "creator_id": item_creator_ids.astype(np.int64),
            "duration_sec": item_durations,
            "title": item_titles,
            "quality": item_quality,
            "pop_rank": item_pop_rank,
        }
    )

    # Pre-index creators by primary topic for user favorite creator assignment
    topic_to_creators: list[np.ndarray] = [
        np.where(creator_primary_topic == t)[0] for t in range(num_topics)
    ]

    # 3. Users: sparse multi-topic preference vector, favorite creators, duration preference
    user_topic_prefs = np.zeros((num_users, num_topics), dtype=np.float32)
    user_pref_log_dur = rng.normal(np.log(24.0), 0.45, size=num_users).astype(np.float32)
    user_like_bias = rng.normal(0.0, 0.25, size=num_users).astype(np.float32)
    user_fav_creators: dict[int, set[int]] = {}
    user_creator_mask_rows: list[np.ndarray] = []

    for u in range(num_users):
        k_topics = int(rng.integers(2, 5))
        chosen_topics = rng.choice(num_topics, size=k_topics, replace=False)
        weights = rng.dirichlet(np.full(k_topics, 1.5))
        user_topic_prefs[u, chosen_topics] = weights.astype(np.float32)

        fav_pool: list[int] = []
        for t in chosen_topics:
            cand = topic_to_creators[int(t)]
            if len(cand) > 0:
                take = min(len(cand), int(rng.integers(1, 3)))
                fav_pool.extend(rng.choice(cand, size=take, replace=False).tolist())
        fav_set = set(fav_pool)
        user_fav_creators[u] = fav_set
        user_creator_mask_rows.append(np.array(sorted(fav_set), dtype=np.int64))

    # 4. Vectorized chunked chronological interaction simulation
    base_timestamp = 1_735_689_600  # 2025-01-01 00:00:00 UTC
    horizon_sec = 30 * 24 * 3600  # 30 days

    n_per_user = cfg.interactions_per_user
    n_pop_channel = int(round(n_per_user * cfg.popularity_exposure_mix))
    n_disc_channel = n_per_user - n_pop_channel

    all_users: list[np.ndarray] = []
    all_items: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_watch_ratio: list[np.ndarray] = []
    all_liked: list[np.ndarray] = []
    all_skipped: list[np.ndarray] = []

    # Mildly flattened popularity weights for discovery channel
    disc_base_w = (zipf_weights ** 0.20)
    disc_base_w /= disc_base_w.sum()

    chunk_size = 200
    for start_u in range(0, num_users, chunk_size):
        end_u = min(start_u + chunk_size, num_users)
        bs = end_u - start_u

        # Topic affinity matrix for chunk: [bs, num_items]
        chunk_topic_aff = user_topic_prefs[start_u:end_u, item_topic_ids]  # float32

        # Creator affinity mask for chunk: [bs, num_items]
        chunk_creator_aff = np.zeros((bs, num_items), dtype=np.float32)
        for idx_in_chunk, u in enumerate(range(start_u, end_u)):
            fav_c = user_creator_mask_rows[u]
            if fav_c.size > 0:
                chunk_creator_aff[idx_in_chunk] = np.isin(item_creator_ids, fav_c).astype(np.float32)

        # Two-channel feed exposure probabilities:
        # Channel A (Viral/Popular Feed): heavy Zipfian skew + moderate topic matching
        pop_probs = zipf_weights[None, :] * (0.30 + 3.2 * chunk_topic_aff + 1.2 * chunk_creator_aff)
        pop_probs /= pop_probs.sum(axis=1, keepdims=True)

        # Channel B (Interest Discovery Feed): strong topic/creator matching across mid & long tail
        disc_probs = disc_base_w[None, :] * (0.04 + 7.5 * chunk_topic_aff + 9.0 * chunk_creator_aff)
        disc_probs /= disc_probs.sum(axis=1, keepdims=True)

        # Combined exposure mixture (sampling without replacement per user -> realistic unique Reels watches)
        mix_probs = cfg.popularity_exposure_mix * pop_probs + (1.0 - cfg.popularity_exposure_mix) * disc_probs
        mix_probs /= mix_probs.sum(axis=1, keepdims=True)

        # Fast vectorized weighted sampling without replacement using Gumbel-top-k trick!
        # Key: log(p) + Gumbel(0, 1) -> argpartition top n_per_user
        gumbel_noise = -np.log(-np.log(rng.uniform(1e-12, 1.0 - 1e-12, size=(bs, num_items))))
        keys = np.log(np.clip(mix_probs, 1e-15, 1.0)) + gumbel_noise
        sampled_items = np.argpartition(keys, -n_per_user, axis=1)[:, -n_per_user:]

        # Randomize chronological order of exposed videos per user
        for idx_in_chunk in range(bs):
            rng.shuffle(sampled_items[idx_in_chunk])

        # Compute true user-video relevance on sampled items: [bs, n_per_user]
        row_idx = np.arange(bs)[:, None]
        sel_topic_aff = chunk_topic_aff[row_idx, sampled_items]
        sel_creator_aff = chunk_creator_aff[row_idx, sampled_items]
        sel_log_dur = item_log_dur[sampled_items]
        sel_quality = item_quality[sampled_items]
        dur_penalty = np.abs(sel_log_dur - user_pref_log_dur[start_u:end_u, None])

        # True relevance combines topic affinity, creator affinity, duration match, intrinsic quality
        eps = rng.normal(0.0, 0.55, size=(bs, n_per_user)).astype(np.float32)
        relevance = (
            4.2 * sel_topic_aff
            + 2.1 * sel_creator_aff
            - 0.35 * dur_penalty
            + 0.55 * sel_quality
            + user_like_bias[start_u:end_u, None]
            + eps
        )

        # Watch completion ratio (realistic short-video replays up to 1.9x)
        watch_noise = rng.normal(0.0, 0.12, size=(bs, n_per_user)).astype(np.float32)
        watch_ratio = np.clip(0.12 + 1.55 * _sigmoid(0.95 * relevance - 0.85) + watch_noise, 0.05, 1.95)

        # Like probability driven by true relevance (~30-38% overall like rate)
        like_prob = _sigmoid(1.35 * relevance - 1.85)
        liked = (rng.random(size=(bs, n_per_user)) < like_prob).astype(np.int8)
        skipped = ((watch_ratio < 0.38) & (liked == 0)).astype(np.int8)

        # Strictly increasing per-user timestamps over 30-day horizon
        ts_offsets = rng.integers(0, horizon_sec, size=(bs, n_per_user))
        ts_offsets.sort(axis=1)
        # Ensure strictly non-decreasing + tie-break jitter
        timestamps = base_timestamp + ts_offsets

        user_col = np.repeat(np.arange(start_u, end_u, dtype=np.int64), n_per_user)
        all_users.append(user_col)
        all_items.append(sampled_items.reshape(-1).astype(np.int64))
        all_timestamps.append(timestamps.reshape(-1).astype(np.int64))
        all_watch_ratio.append(watch_ratio.reshape(-1).astype(np.float32))
        all_liked.append(liked.reshape(-1).astype(np.int8))
        all_skipped.append(skipped.reshape(-1).astype(np.int8))

    interactions_df = pd.DataFrame(
        {
            "user_id": np.concatenate(all_users),
            "item_id": np.concatenate(all_items),
            "timestamp": np.concatenate(all_timestamps),
            "watch_ratio": np.concatenate(all_watch_ratio),
            "liked": np.concatenate(all_liked),
            "skipped": np.concatenate(all_skipped),
        }
    )

    return interactions_df, items_meta_df, topic_names, creator_names, user_fav_creators


def load_kuairec_interactions(
    cfg: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[int, set[int]]]:
    """Load real KuaiRec 2.0 CSV files from `cfg.data_dir`.

    Expected files in `data_dir` (standard KuaiRec 2.0 layout):
      - `big_matrix.csv` or `small_matrix.csv` (required):
          user_id, video_id, play_duration, video_duration, time, date, timestamp, watch_ratio
      - `item_categories.csv` (optional): video_id, feat (e.g. "[12, 3]")
      - `item_daily_features.csv` or `video_features.csv` (optional): video_id, author_id, video_duration
    """
    if not cfg.data_dir:
        raise ValueError(
            "KuaiRec mode requires `--data-dir <path>` pointing to extracted KuaiRec 2.0 CSV files "
            "(big_matrix.csv or small_matrix.csv). Download from https://kuairec.com/ or "
            "https://zenodo.org/records/10439422."
        )

    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"KuaiRec data directory not found: {data_dir}. "
            "Download KuaiRec 2.0 (big_matrix.csv / small_matrix.csv, item_categories.csv) "
            "from https://kuairec.com/ and extract into this folder."
        )

    matrix_path = None
    for candidate in ("small_matrix.csv", "big_matrix.csv", "kuairec.csv"):
        p = data_dir / candidate
        if p.exists():
            matrix_path = p
            break
    if matrix_path is None:
        raise FileNotFoundError(
            f"No KuaiRec matrix CSV (small_matrix.csv / big_matrix.csv) found in {data_dir}."
        )

    raw_df = pd.read_csv(matrix_path, nrows=cfg.max_kuairec_rows)
    required_cols = {"user_id", "video_id", "watch_ratio"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"KuaiRec CSV {matrix_path} missing required columns: {sorted(missing)}")

    # Remap arbitrary user_id / video_id to contiguous [0..U-1] and [0..I-1]
    raw_df = raw_df.dropna(subset=["user_id", "video_id", "watch_ratio"]).copy()
    if "timestamp" in raw_df.columns:
        ts = pd.to_numeric(raw_df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    elif "time" in raw_df.columns:
        ts = pd.to_datetime(raw_df["time"], errors="coerce").astype("int64") // 10**9
        ts = ts.fillna(0).astype(np.int64)
    else:
        ts = np.arange(len(raw_df), dtype=np.int64)
    raw_df["timestamp"] = ts

    unique_users, user_idx = np.unique(raw_df["user_id"].to_numpy(), return_inverse=True)
    unique_items, item_idx = np.unique(raw_df["video_id"].to_numpy(), return_inverse=True)
    raw_item_to_idx = {int(orig): int(i) for i, orig in enumerate(unique_items)}

    watch_ratio = np.clip(pd.to_numeric(raw_df["watch_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, 5.0)
    liked = (watch_ratio >= cfg.kuairec_like_threshold).astype(np.int8)
    skipped = (watch_ratio < cfg.kuairec_skip_threshold).astype(np.int8)

    interactions_df = pd.DataFrame(
        {
            "user_id": user_idx.astype(np.int64),
            "item_id": item_idx.astype(np.int64),
            "timestamp": raw_df["timestamp"].to_numpy(dtype=np.int64),
            "watch_ratio": watch_ratio,
            "liked": liked,
            "skipped": skipped,
        }
    )

    num_items = len(unique_items)
    # Extract video duration from matrix if present
    if "video_duration" in raw_df.columns:
        dur_series = (
            raw_df.groupby("video_id")["video_duration"].median().reindex(unique_items).fillna(25000.0).to_numpy(dtype=np.float32)
        )
        # KuaiRec video_duration is in milliseconds
        if np.median(dur_series) > 500.0:
            dur_series = dur_series / 1000.0
        item_durations = np.clip(dur_series, 3.0, 300.0).astype(np.float32)
    else:
        item_durations = np.full(num_items, 25.0, dtype=np.float32)

    # Parse item_categories.csv if present
    item_topic_ids = np.zeros(num_items, dtype=np.int64)
    cat_path = data_dir / "item_categories.csv"
    if cat_path.exists():
        cat_df = pd.read_csv(cat_path)
        if {"video_id", "feat"}.issubset(cat_df.columns):
            raw_topics: list[int] = []
            mapped_pairs: list[tuple[int, int]] = []
            for vid, feat_str in zip(cat_df["video_id"], cat_df["feat"]):
                if int(vid) not in raw_item_to_idx:
                    continue
                try:
                    parsed = ast.literal_eval(str(feat_str))
                    first_tag = int(parsed[0]) if isinstance(parsed, (list, tuple)) and len(parsed) > 0 else int(parsed)
                except (ValueError, SyntaxError, TypeError):
                    first_tag = 0
                raw_topics.append(first_tag)
                mapped_pairs.append((raw_item_to_idx[int(vid)], first_tag))
            unique_topics = sorted(set(raw_topics)) if raw_topics else [0]
            topic_remap = {t: i for i, t in enumerate(unique_topics)}
            for i_idx, t_raw in mapped_pairs:
                item_topic_ids[i_idx] = topic_remap[t_raw]
            topic_names = [f"Category_{t}" for t in unique_topics]
        else:
            topic_names = ["Category_0"]
    else:
        topic_names = ["Category_0"]

    # Parse creator/author IDs if item_daily_features.csv or video_features.csv exists
    item_creator_ids = np.zeros(num_items, dtype=np.int64)
    for feat_fname in ("item_daily_features.csv", "video_features.csv"):
        feat_path = data_dir / feat_fname
        if feat_path.exists():
            fdf = pd.read_csv(feat_path, usecols=lambda c: c in {"video_id", "author_id"})
            if {"video_id", "author_id"}.issubset(fdf.columns):
                fdf = fdf.drop_duplicates(subset=["video_id"])
                author_map = dict(zip(fdf["video_id"].astype(int), fdf["author_id"].astype(int)))
                raw_authors = [author_map.get(int(orig), 0) for orig in unique_items]
                unique_authors, creator_idx = np.unique(raw_authors, return_inverse=True)
                item_creator_ids = creator_idx.astype(np.int64)
                creator_names = [f"@author_{a}" for a in unique_authors]
                break
    else:
        # Deterministic fallback bucket if author metadata file is unavailable
        num_buckets = min(200, max(1, num_items // 10))
        item_creator_ids = (np.arange(num_items) % num_buckets).astype(np.int64)
        creator_names = [f"@creator_bucket_{c:03d}" for c in range(num_buckets)]

    # Compute popularity rank from raw frequency for metadata display
    raw_counts = np.bincount(item_idx, minlength=num_items)
    pop_order = np.argsort(-raw_counts)
    pop_rank = np.empty(num_items, dtype=np.int64)
    pop_rank[pop_order] = np.arange(1, num_items + 1)

    items_meta_df = pd.DataFrame(
        {
            "item_id": np.arange(num_items, dtype=np.int64),
            "topic_id": item_topic_ids,
            "creator_id": item_creator_ids,
            "duration_sec": item_durations,
            "title": [f"KuaiRec Video {orig}" for orig in unique_items],
            "quality": np.zeros(num_items, dtype=np.float32),
            "pop_rank": pop_rank,
        }
    )
    return interactions_df, items_meta_df, topic_names, creator_names, {}


def chronological_per_user_split(
    interactions_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological per-user train/val/test split with ZERO temporal leakage.

    For each user, interactions are sorted strictly ascending by timestamp.
    The earliest `train_ratio` go to train, next `val_ratio` to val, remainder to test.
    """
    df = interactions_df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    user_counts = df.groupby("user_id")["timestamp"].transform("count").to_numpy()
    pos_in_user = df.groupby("user_id").cumcount().to_numpy()

    train_cut = np.floor(user_counts * train_ratio).astype(np.int64)
    val_cut = np.floor(user_counts * (train_ratio + val_ratio)).astype(np.int64)
    # Guarantee at least 1 train row for users with >= 3 interactions
    train_cut = np.maximum(train_cut, np.minimum(1, user_counts))
    val_cut = np.maximum(val_cut, train_cut)

    train_mask = pos_in_user < train_cut
    val_mask = (pos_in_user >= train_cut) & (pos_in_user < val_cut)
    test_mask = pos_in_user >= val_cut

    return (
        df.loc[train_mask].reset_index(drop=True),
        df.loc[val_mask].reset_index(drop=True),
        df.loc[test_mask].reset_index(drop=True),
    )


def extract_train_only_features(
    train_df: pd.DataFrame,
    items_meta_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    num_topics: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract user dense features, item dense features, and item popularity strictly from `train_df`.

    Returns:
        user_dense: [num_users, 5 + num_topics] float32
        item_dense: [num_items, 6] float32
        item_train_counts: [num_items] int64
        item_pos_counts: [num_items] int64
        item_probs: [num_items] float32 (smoothed p_item for Log-Q)
        log_item_probs: [num_items] float32
    """
    item_topics = items_meta_df["topic_id"].to_numpy(dtype=np.int64)
    item_durations = items_meta_df["duration_sec"].to_numpy(dtype=np.float32)
    item_log_dur = np.log(np.clip(item_durations, 1.0, None))
    dur_mean = float(item_log_dur.mean())
    dur_std = float(item_log_dur.std() + 1e-6)
    norm_log_dur = (item_log_dur - dur_mean) / dur_std

    u_ids = train_df["user_id"].to_numpy(dtype=np.int64)
    i_ids = train_df["item_id"].to_numpy(dtype=np.int64)
    wr = train_df["watch_ratio"].to_numpy(dtype=np.float32)
    liked = train_df["liked"].to_numpy(dtype=np.float32)
    skipped = train_df["skipped"].to_numpy(dtype=np.float32)

    # ---- User Features (strictly from train_df) ----
    u_counts = np.bincount(u_ids, minlength=num_users).astype(np.float32)
    u_counts_safe = np.maximum(u_counts, 1.0)
    u_wr_sum = np.bincount(u_ids, weights=wr, minlength=num_users).astype(np.float32)
    u_like_sum = np.bincount(u_ids, weights=liked, minlength=num_users).astype(np.float32)
    u_skip_sum = np.bincount(u_ids, weights=skipped, minlength=num_users).astype(np.float32)

    global_wr = float(wr.mean()) if len(wr) else 0.5
    global_like = float(liked.mean()) if len(liked) else 0.25
    global_skip = float(skipped.mean()) if len(skipped) else 0.30

    # Bayesian-smoothed user completion, like, skip rates
    prior_u = 3.0
    u_comp_rate = (u_wr_sum + prior_u * global_wr) / (u_counts + prior_u)
    u_like_rate = (u_like_sum + prior_u * global_like) / (u_counts + prior_u)
    u_skip_rate = (u_skip_sum + prior_u * global_skip) / (u_counts + prior_u)

    # Preferred normalized duration from positive/engaged watches
    eng_weight = np.maximum(2.0 * liked + 0.4 * np.clip(wr, 0.0, 1.5) - 0.3 * skipped, 0.05)
    u_dur_num = np.bincount(u_ids, weights=eng_weight * norm_log_dur[i_ids], minlength=num_users).astype(np.float32)
    u_dur_den = np.bincount(u_ids, weights=eng_weight, minlength=num_users).astype(np.float32)
    u_pref_dur = np.where(u_dur_den > 0, u_dur_num / np.maximum(u_dur_den, 1e-6), 0.0).astype(np.float32)

    # User topic preference vector [num_users, num_topics] from train interactions
    topic_weights = np.maximum(2.5 * liked + 0.5 * np.clip(wr, 0.0, 1.5) - 0.4 * skipped, 0.0)
    flat_idx = u_ids * num_topics + item_topics[i_ids]
    u_topic_vec = np.bincount(flat_idx, weights=topic_weights, minlength=num_users * num_topics).reshape(
        num_users, num_topics
    ).astype(np.float32)
    row_sums = u_topic_vec.sum(axis=1, keepdims=True)
    u_topic_vec = np.where(row_sums > 1e-6, u_topic_vec / np.maximum(row_sums, 1e-6), 1.0 / num_topics).astype(
        np.float32
    )

    u_log_act = (np.log1p(u_counts) / 5.0).astype(np.float32)
    user_dense = np.concatenate(
        [
            u_log_act[:, None],
            (u_comp_rate / 2.0)[:, None].astype(np.float32),
            u_like_rate[:, None].astype(np.float32),
            u_skip_rate[:, None].astype(np.float32),
            u_pref_dur[:, None].astype(np.float32),
            u_topic_vec,
        ],
        axis=1,
    ).astype(np.float32)

    # ---- Item Features & Empirical Popularity (strictly from train_df) ----
    i_counts = np.bincount(i_ids, minlength=num_items).astype(np.int64)
    i_pos_counts = np.bincount(i_ids[liked > 0.5], minlength=num_items).astype(np.int64)
    i_wr_sum = np.bincount(i_ids, weights=wr, minlength=num_items).astype(np.float32)
    i_like_sum = np.bincount(i_ids, weights=liked, minlength=num_items).astype(np.float32)
    i_skip_sum = np.bincount(i_ids, weights=skipped, minlength=num_items).astype(np.float32)

    prior_i = 5.0
    i_counts_f = i_counts.astype(np.float32)
    i_comp_rate = (i_wr_sum + prior_i * global_wr) / (i_counts_f + prior_i)
    i_like_rate = (i_like_sum + prior_i * global_like) / (i_counts_f + prior_i)
    i_skip_rate = (i_skip_sum + prior_i * global_skip) / (i_counts_f + prior_i)

    i_log_imp = np.log1p(i_counts_f)
    i_log_imp_z = (i_log_imp - i_log_imp.mean()) / (i_log_imp.std() + 1e-6)
    dur_scaled = np.clip(item_durations / 120.0, 0.0, 2.0)

    item_dense = np.stack(
        [
            norm_log_dur.astype(np.float32),
            dur_scaled.astype(np.float32),
            i_log_imp_z.astype(np.float32),
            (i_comp_rate / 2.0).astype(np.float32),
            i_like_rate.astype(np.float32),
            i_skip_rate.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    # Smoothed empirical positive sampling probability p_item for Log-Q correction
    # Laplace smoothing (+0.5) guarantees strictly positive p_item and finite log(p_item) for cold/rare items
    smoothed_pos = i_pos_counts.astype(np.float64) + 0.5
    item_probs = (smoothed_pos / smoothed_pos.sum()).astype(np.float32)
    log_item_probs = np.log(item_probs).astype(np.float32)

    return user_dense, item_dense, i_counts, i_pos_counts, item_probs, log_item_probs


def _build_user_item_sets(df: pd.DataFrame, only_liked: bool = False) -> dict[int, set[int]]:
    sub = df.loc[df["liked"] == 1] if only_liked else df
    grouped = sub.groupby("user_id")["item_id"].apply(set)
    return {int(u): s for u, s in grouped.items()}


def load_or_generate_dataset(cfg: DataConfig) -> DatasetBundle:
    """Build complete leakage-free DatasetBundle for synthetic or KuaiRec data."""
    if cfg.dataset.lower() == "kuairec":
        interactions_df, items_meta_df, topic_names, creator_names, user_fav_creators = load_kuairec_interactions(cfg)
        dataset_name = "kuairec"
    elif cfg.dataset.lower() == "synthetic":
        interactions_df, items_meta_df, topic_names, creator_names, user_fav_creators = generate_synthetic_interactions(cfg)
        dataset_name = "synthetic_zipfian_reels"
    else:
        raise ValueError(f"Unsupported dataset '{cfg.dataset}'. Expected 'synthetic' or 'kuairec'.")

    num_users = int(interactions_df["user_id"].max()) + 1
    num_items = int(len(items_meta_df))
    num_topics = len(topic_names)
    num_creators = len(creator_names)

    train_df, val_df, test_df = chronological_per_user_split(
        interactions_df,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )

    user_dense, item_dense, item_train_counts, item_pos_counts, item_probs, log_item_probs = (
        extract_train_only_features(
            train_df=train_df,
            items_meta_df=items_meta_df,
            num_users=num_users,
            num_items=num_items,
            num_topics=num_topics,
        )
    )

    train_pos_df = train_df.loc[train_df["liked"] == 1]
    train_pos_users = train_pos_df["user_id"].to_numpy(dtype=np.int64, copy=True)
    train_pos_items = train_pos_df["item_id"].to_numpy(dtype=np.int64, copy=True)

    train_seen = _build_user_item_sets(train_df, only_liked=False)
    val_seen = _build_user_item_sets(val_df, only_liked=False)
    train_val_seen: dict[int, set[int]] = {}
    for u in range(num_users):
        train_val_seen[u] = train_seen.get(u, set()) | val_seen.get(u, set())

    val_liked = _build_user_item_sets(val_df, only_liked=True)
    test_liked = _build_user_item_sets(test_df, only_liked=True)

    # Compute honest dataset statistics (including Zipf head concentration in train)
    sorted_pos = np.sort(item_pos_counts)[::-1]
    total_pos = max(int(sorted_pos.sum()), 1)
    top5_n = max(1, int(round(0.05 * num_items)))
    top10_n = max(1, int(round(0.10 * num_items)))
    stats = {
        "dataset": dataset_name,
        "num_users": num_users,
        "num_items": num_items,
        "num_topics": num_topics,
        "num_creators": num_creators,
        "total_interactions": int(len(interactions_df)),
        "train_interactions": int(len(train_df)),
        "val_interactions": int(len(val_df)),
        "test_interactions": int(len(test_df)),
        "train_positive_pairs": int(len(train_pos_users)),
        "overall_like_rate": float(interactions_df["liked"].mean()),
        "overall_skip_rate": float(interactions_df["skipped"].mean()),
        "train_top5pct_pos_share": float(sorted_pos[:top5_n].sum() / total_pos),
        "train_top10pct_pos_share": float(sorted_pos[:top10_n].sum() / total_pos),
        "val_eval_users": len(val_liked),
        "test_eval_users": len(test_liked),
        "avg_test_liked_per_user": float(np.mean([len(s) for s in test_liked.values()])) if test_liked else 0.0,
    }

    return DatasetBundle(
        dataset_name=dataset_name,
        num_users=num_users,
        num_items=num_items,
        num_topics=num_topics,
        num_creators=num_creators,
        topic_names=topic_names,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_pos_users=train_pos_users,
        train_pos_items=train_pos_items,
        user_dense=np.array(user_dense, dtype=np.float32, copy=True),
        item_dense=np.array(item_dense, dtype=np.float32, copy=True),
        item_topic_ids=items_meta_df["topic_id"].to_numpy(dtype=np.int64, copy=True),
        item_creator_ids=items_meta_df["creator_id"].to_numpy(dtype=np.int64, copy=True),
        item_durations=items_meta_df["duration_sec"].to_numpy(dtype=np.float32, copy=True),
        item_train_counts=np.array(item_train_counts, dtype=np.int64, copy=True),
        item_pos_counts=np.array(item_pos_counts, dtype=np.int64, copy=True),
        item_probs=np.array(item_probs, dtype=np.float32, copy=True),
        log_item_probs=np.array(log_item_probs, dtype=np.float32, copy=True),
        val_liked=val_liked,
        test_liked=test_liked,
        train_seen=train_seen,
        train_val_seen=train_val_seen,
        item_titles=items_meta_df["title"].tolist(),
        creator_names=creator_names,
        user_favorite_creators=user_fav_creators,
        stats=stats,
    )
