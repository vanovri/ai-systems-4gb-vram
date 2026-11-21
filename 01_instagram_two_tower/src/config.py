"""Configuration dataclasses for the Two-Tower Instagram Reels RecSys pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TOPICS: tuple[str, ...] = (
    "Comedy",
    "Dance",
    "Tech & AI",
    "Cooking",
    "Fitness",
    "Travel",
    "Gaming",
    "Fashion",
    "Pets",
    "Music",
    "DIY & Crafts",
    "Photography",
    "Personal Finance",
    "Cars & Moto",
    "Sports",
    "Digital Art",
    "Education",
    "Beauty & Care",
    "Movies & TV",
    "Science",
    "Outdoors",
    "Lifestyle",
    "Architecture",
    "Books & Writing",
)


@dataclass
class DataConfig:
    dataset: str = "synthetic"  # "synthetic" | "kuairec"
    data_dir: str | None = None
    num_users: int = 2000
    num_items: int = 5000
    num_creators: int = 400
    num_topics: int = len(TOPICS)
    interactions_per_user: int = 115  # 2000 * 115 = 230,000 interactions
    zipf_exponent: float = 1.20
    popularity_exposure_mix: float = 0.78
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    kuairec_like_threshold: float = 1.0
    kuairec_skip_threshold: float = 0.3
    max_kuairec_rows: int | None = None
    seed: int = 42


@dataclass
class ModelConfig:
    embed_dim: int = 64
    hidden_dim: int = 192
    proj_hidden_dim: int = 96
    dropout: float = 0.25
    temperature: float = 0.05  # tau = 0.05 per Two-Tower spec
    user_id_emb_dim: int = 16
    item_id_emb_dim: int = 16
    topic_emb_dim: int = 16
    creator_emb_dim: int = 24


@dataclass
class TrainConfig:
    variant: str = "baseline"  # "baseline" | "debiased"
    epochs: int = 14
    batch_size: int = 512
    lr: float = 1.2e-3
    weight_decay: float = 5e-3
    grad_clip: float = 2.0
    # Log-Q popularity debiasing parameters
    logq_alpha: float = 1.0  # Training-time log-Q sampling bias correction (Yi et al. 2019)
    infer_logq_beta: float = 0.35  # Inference-time serving popularity debiasing
    # Hard + uniform negative mixing for debiased variant
    extra_neg_count: int = 64
    hard_neg_pool_factor: int = 4
    hard_neg_fraction: float = 0.5
    use_amp_fp16: bool = True  # GTX 1650 Ti SM 7.5: FP16 only, NEVER BF16
    eval_every: int = 2
    seed: int = 42
    device: str = "auto"


@dataclass
class EvalConfig:
    k_values: tuple[int, ...] = (10, 20, 50)
    head_item_fractions: tuple[float, ...] = (0.01, 0.05, 0.10)
    latency_queries: int = 500
    latency_warmup: int = 50
    batch_size: int = 256


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_paths(self, root: Path) -> tuple[Path, Path]:
        art = Path(self.artifacts_dir)
        rep = Path(self.reports_dir)
        if not art.is_absolute():
            art = root / art
        if not rep.is_absolute():
            rep = root / rep
        art.mkdir(parents=True, exist_ok=True)
        rep.mkdir(parents=True, exist_ok=True)
        return art, rep
