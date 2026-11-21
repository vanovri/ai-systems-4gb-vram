"""End-to-end smoke test: full baseline vs debiased training, Log-Q debiasing effect, report artifacts, and demo CLI."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.demo import predict
from src.train import run_benchmark


def test_e2e_smoke_train_eval_logq_debiasing_and_demo(tmp_path: Path) -> None:
    exp_cfg = ExperimentConfig(
        data=DataConfig(
            dataset="synthetic",
            num_users=320,
            num_items=900,
            num_creators=90,
            interactions_per_user=60,
            zipf_exponent=1.25,
            popularity_exposure_mix=0.80,
            seed=42,
        ),
        model=ModelConfig(embed_dim=64, hidden_dim=128, proj_hidden_dim=64, dropout=0.2, temperature=0.05),
        train=TrainConfig(
            epochs=5,
            batch_size=128,
            lr=1.5e-3,
            weight_decay=3e-3,
            logq_alpha=1.0,
            infer_logq_beta=0.35,
            extra_neg_count=32,
            eval_every=2,
            use_amp_fp16=False,
            device="cpu",
            seed=42,
        ),
        eval=EvalConfig(k_values=(10, 20, 50), latency_queries=40, latency_warmup=5, batch_size=128),
        artifacts_dir=str(tmp_path / "artifacts"),
        reports_dir=str(tmp_path / "reports"),
    )

    run_benchmark(exp_cfg, variants=("baseline", "debiased"), project_root=tmp_path, verbose=False)

    # Verify generated files exist and are non-empty
    bench_json = tmp_path / "reports" / "benchmark.json"
    plot_png = tmp_path / "reports" / "comparison_plot.png"
    assert bench_json.exists() and bench_json.stat().st_size > 100
    assert plot_png.exists() and plot_png.stat().st_size > 5000
    assert (tmp_path / "artifacts" / "baseline_best.pt").exists()
    assert (tmp_path / "artifacts" / "debiased_best.pt").exists()

    parsed = json.loads(bench_json.read_text(encoding="utf-8"))
    base_m = parsed["variants"]["baseline"]["test_metrics"]
    deb_m = parsed["variants"]["debiased"]["test_metrics"]
    logq0_m = parsed["variants"]["debiased"]["logq_train_only_test_metrics"]

    # 1. Training Log-Q sampling-bias correction (Yi et al. 2019) strongly improves ranking over uncorrected in-batch baseline
    assert logq0_m["recall@10"] > base_m["recall@10"]
    assert deb_m["recall@10"] > base_m["recall@10"]
    assert deb_m["ndcg@10"] > base_m["ndcg@10"]

    # 2. Serving-time Log-Q popularity debiasing (beta > 0 vs beta = 0) reduces viral head dominance,
    #    lowers Popularity Gini, and boosts Catalog Coverage@10 & Long-Tail Share
    assert deb_m["head_item_share_top5pct@10"] < logq0_m["head_item_share_top5pct@10"]
    assert deb_m["popularity_gini@10"] < logq0_m["popularity_gini@10"]
    assert deb_m["coverage@10"] > logq0_m["coverage@10"]
    assert deb_m["long_tail_share_bottom80pct@10"] > logq0_m["long_tail_share_bottom80pct@10"]

    # 3. Verify demo `predict(user_id, top_k=10)` works end-to-end on saved checkpoints
    recs = predict(user_id=0, top_k=10, variant="debiased", artifacts_dir=tmp_path / "artifacts", device="cpu")
    assert len(recs) == 10
    assert [r["rank"] for r in recs] == list(range(1, 11))
    assert all(-1.0001 <= r["cosine_score"] <= 1.0001 for r in recs)
    assert all(isinstance(r["match_reason"], str) and len(r["match_reason"]) > 5 for r in recs)
    scores = [r["ranking_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
