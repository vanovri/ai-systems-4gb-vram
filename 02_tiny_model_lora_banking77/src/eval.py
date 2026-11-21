"""Head-to-head Banking77 evaluation: Qwen3-0.6B Base (Zero-Shot & 5-Shot) vs Qwen3-0.6B + LoRA.

Reports strictly real measured local metrics:
- Exact-Match Accuracy (Top-1)
- Macro-F1 (across all 77 banking intents; invalid predictions penalized as -1)
- Top-3 Accuracy (beam search num_beams=3, num_return_sequences=3)
- Invalid-Label Rate
- Peak VRAM (MiB)
- Batch Throughput (samples/sec)
- Single-Sample Interactive Latency p50 / p99 (ms)
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

_ORIG_LINEAR = F.linear


@contextlib.contextmanager
def tu117_fast_inference_ctx(device: str = "cuda"):
    """Accelerates FP16-stored transformer Linear ops on Tensor-Core-less GTX 16xx (TU116/TU117)."""
    use_fast = (
        device == "cuda"
        and torch.cuda.is_available()
        and ("1650" in torch.cuda.get_device_name(0) or "1660" in torch.cuda.get_device_name(0))
    )
    if not use_fast:
        yield
        return

    def _fast_linear(inp, weight, bias=None):
        if inp.is_cuda and weight.dtype == torch.float16 and weight.shape[0] <= 16384:
            out = _ORIG_LINEAR(inp.float(), weight.float(), bias.float() if bias is not None else None)
            return out.to(inp.dtype)
        return _ORIG_LINEAR(inp, weight, bias)

    F.linear = _fast_linear
    try:
        yield
    finally:
        F.linear = _ORIG_LINEAR

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import (  # noqa: E402
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    build_few_shot_prompt,
    build_prompt,
    load_banking77_splits,
    parse_predicted_label,
)


def select_eval_subset(
    records: Sequence[Dict[str, Any]],
    num_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Return full test set if num_samples <= 0 or >= len(records), else stratified subset across all 77 intents."""
    if num_samples <= 0 or num_samples >= len(records):
        return list(records)
    y = [r["label"] for r in records]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=num_samples, random_state=seed)
    _, subset_idx = next(splitter.split(records, y))
    return [records[i] for i in sorted(subset_idx)]


def make_beam_gen_config(
    tokenizer: Any,
    max_new_tokens: int = 16,
    num_beams: int = 3,
    num_return_sequences: int = 3,
) -> GenerationConfig:
    """Clean deterministic beam-search GenerationConfig avoiding Qwen3 sampling-flag warnings."""
    return GenerationConfig(
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
    )


def compute_classification_metrics(
    gold_labels: Sequence[str],
    top1_preds: Sequence[str],
    top3_preds: Sequence[Sequence[str]],
    label2id: Dict[str, int],
) -> Dict[str, Any]:
    """Compute Exact-Match Acc, Macro-F1 (77 classes), Top-3 Acc, and Invalid-Label Rate."""
    n = len(gold_labels)
    if n == 0:
        raise ValueError("Cannot compute metrics on empty prediction list")

    valid_set = set(label2id.keys())
    exact_hits = 0
    top3_hits = 0
    invalid_count = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for gold, pred1, cands3 in zip(gold_labels, top1_preds, top3_preds):
        y_true.append(label2id[gold])
        if pred1 in valid_set:
            y_pred.append(label2id[pred1])
        else:
            invalid_count += 1
            y_pred.append(-1)

        if pred1 == gold:
            exact_hits += 1
        if gold in cands3:
            top3_hits += 1

    all_class_ids = sorted(label2id.values())
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=all_class_ids, average="macro", zero_division=0)
    )
    return {
        "num_samples": n,
        "exact_match_accuracy": round(exact_hits / n, 4),
        "exact_match_pct": round(100.0 * exact_hits / n, 2),
        "macro_f1": round(macro_f1, 4),
        "macro_f1_pct": round(100.0 * macro_f1, 2),
        "top3_accuracy": round(top3_hits / n, 4),
        "top3_pct": round(100.0 * top3_hits / n, 2),
        "invalid_label_count": invalid_count,
        "invalid_label_rate": round(invalid_count / n, 4),
        "invalid_label_pct": round(100.0 * invalid_count / n, 2),
    }


@torch.no_grad()
def evaluate_model_on_records(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    label2id: Dict[str, int],
    prompt_mode: str = "standard",
    batch_size: int = 16,
    max_new_tokens: int = 16,
    num_beams: int = 3,
    latency_samples: int = 40,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run batched beam-search evaluation + single-sample p50/p99 latency benchmark."""
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    valid_labels = list(label2id.keys())
    prompt_fn = build_few_shot_prompt if prompt_mode == "few_shot" else build_prompt
    prompts = [prompt_fn(r["text"]) for r in records]
    gold_labels = [r["label_text"] for r in records]

    gen_cfg = make_beam_gen_config(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
    )

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    top1_preds: List[str] = []
    top3_preds: List[List[str]] = []
    sample_outputs: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    with tu117_fast_inference_ctx(device):
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256 if prompt_mode == "few_shot" else 128,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out = model.generate(
                    **enc,
                    generation_config=gen_cfg,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            gen_tokens = out.sequences[:, prompt_len:]
            decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

            for i in range(len(batch_prompts)):
                raw_cands = decoded[i * num_beams : (i + 1) * num_beams]
                parsed_cands = [parse_predicted_label(c, valid_labels=valid_labels) for c in raw_cands]
                # Deduplicate preserving beam rank
                dedup: List[str] = []
                for p_lbl in parsed_cands:
                    if p_lbl not in dedup:
                        dedup.append(p_lbl)
                pred1 = parsed_cands[0] if parsed_cands else ""
                top1_preds.append(pred1)
                top3_preds.append(dedup[:3])

                global_idx = start + i
                if global_idx < 10:
                    sample_outputs.append(
                        {
                            "query": records[global_idx]["text"],
                            "gold": gold_labels[global_idx],
                            "pred_top1": pred1,
                            "pred_top3": dedup[:3],
                            "raw_top1": raw_cands[0].strip()[:80],
                            "correct": pred1 == gold_labels[global_idx],
                        }
                    )

    if device == "cuda":
        torch.cuda.synchronize()
    total_batch_sec = time.perf_counter() - t0
    peak_vram_mib = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    )

    # Single-sample interactive latency benchmark (p50 / p99) with greedy single-beam decode
    greedy_cfg = make_beam_gen_config(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        num_return_sequences=1,
    )
    lat_n = min(latency_samples, len(prompts))
    latencies_ms: List[float] = []
    with tu117_fast_inference_ctx(device):
        # 3 warmup iterations
        for w_idx in range(min(3, lat_n)):
            w_enc = tokenizer([prompts[w_idx]], return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                _ = model.generate(**w_enc, generation_config=greedy_cfg)
        if device == "cuda":
            torch.cuda.synchronize()

        for idx in range(lat_n):
            s_enc = tokenizer([prompts[idx]], return_tensors="pt", truncation=True, max_length=256).to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            ts = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                _ = model.generate(**s_enc, generation_config=greedy_cfg)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - ts) * 1000.0)

    tokenizer.padding_side = orig_padding_side

    cls_metrics = compute_classification_metrics(
        gold_labels=gold_labels,
        top1_preds=top1_preds,
        top3_preds=top3_preds,
        label2id=label2id,
    )
    cls_metrics.update(
        {
            "prompt_mode": prompt_mode,
            "batch_size": batch_size,
            "num_beams": num_beams,
            "eval_seconds": round(total_batch_sec, 2),
            "throughput_samples_per_sec": round(len(records) / max(total_batch_sec, 1e-6), 2),
            "peak_vram_mib": round(peak_vram_mib, 1),
            "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0,
            "latency_p99_ms": round(float(np.percentile(latencies_ms, 99)), 2) if latencies_ms else 0.0,
            "latency_bench_samples": lat_n,
            "sample_predictions": sample_outputs,
        }
    )
    return cls_metrics


def run_head_to_head(args: argparse.Namespace) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_dir = Path(args.adapter_dir)
    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    splits = load_banking77_splits(val_ratio=0.10, seed=args.seed, dataset_id=args.dataset_id)
    eval_records = select_eval_subset(splits.test, num_samples=args.num_samples, seed=args.seed)
    print(
        f"[Eval] Running head-to-head on {len(eval_records)} Banking77 test samples "
        f"(total test split = {len(splits.test)}, 77 intents)",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load Base Qwen3-0.6B once and evaluate Zero-Shot & 5-Shot
    print("[Eval] Loading base Qwen/Qwen3-0.6B (FP16)...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(device)
    base_model.eval()

    print("[Eval] Evaluating Base (Zero-Shot, same prompt)...", flush=True)
    base_zs = evaluate_model_on_records(
        model=base_model,
        tokenizer=tokenizer,
        records=eval_records,
        label2id=splits.label2id,
        prompt_mode="standard",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=3,
        latency_samples=args.latency_samples,
        device=device,
    )
    print(
        f"  Base Zero-Shot: Acc={base_zs['exact_match_pct']}% | Macro-F1={base_zs['macro_f1_pct']}% | "
        f"Top3={base_zs['top3_pct']}% | Invalid={base_zs['invalid_label_pct']}% | "
        f"p50={base_zs['latency_p50_ms']}ms",
        flush=True,
    )

    print("[Eval] Evaluating Base (5-Shot, snake_case examples)...", flush=True)
    base_fs = evaluate_model_on_records(
        model=base_model,
        tokenizer=tokenizer,
        records=eval_records,
        label2id=splits.label2id,
        prompt_mode="few_shot",
        batch_size=max(args.batch_size // 2, 4),
        max_new_tokens=args.max_new_tokens,
        num_beams=3,
        latency_samples=args.latency_samples,
        device=device,
    )
    print(
        f"  Base 5-Shot:    Acc={base_fs['exact_match_pct']}% | Macro-F1={base_fs['macro_f1_pct']}% | "
        f"Top3={base_fs['top3_pct']}% | Invalid={base_fs['invalid_label_pct']}% | "
        f"p50={base_fs['latency_p50_ms']}ms",
        flush=True,
    )

    # 2. Attach trained LoRA adapter & merge in-memory for zero-overhead inference evaluation
    print(f"[Eval] Loading LoRA adapter from {adapter_dir}...", flush=True)
    lora_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    lora_model = lora_model.merge_and_unload()
    lora_model.eval()

    print("[Eval] Evaluating Qwen3-0.6B + LoRA Fine-Tuned...", flush=True)
    lora_res = evaluate_model_on_records(
        model=lora_model,
        tokenizer=tokenizer,
        records=eval_records,
        label2id=splits.label2id,
        prompt_mode="standard",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=3,
        latency_samples=args.latency_samples,
        device=device,
    )
    print(
        f"  LoRA Fine-Tuned: Acc={lora_res['exact_match_pct']}% | Macro-F1={lora_res['macro_f1_pct']}% | "
        f"Top3={lora_res['top3_pct']}% | Invalid={lora_res['invalid_label_pct']}% | "
        f"p50={lora_res['latency_p50_ms']}ms | p99={lora_res['latency_p99_ms']}ms",
        flush=True,
    )

    # Also evaluate LoRA on full 3,076 test split if eval_records was a subset and --full-lora is enabled
    lora_full_res: Optional[Dict[str, Any]] = None
    if args.full_lora and len(eval_records) < len(splits.test):
        print(f"[Eval] Evaluating LoRA on FULL {len(splits.test)} test samples...", flush=True)
        lora_full_res = evaluate_model_on_records(
            model=lora_model,
            tokenizer=tokenizer,
            records=splits.test,
            label2id=splits.label2id,
            prompt_mode="standard",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=3,
            latency_samples=args.latency_samples,
            device=device,
        )
        print(
            f"  LoRA FULL ({len(splits.test)}): Acc={lora_full_res['exact_match_pct']}% | "
            f"Macro-F1={lora_full_res['macro_f1_pct']}% | Top3={lora_full_res['top3_pct']}% | "
            f"Invalid={lora_full_res['invalid_label_pct']}%",
            flush=True,
        )

    results: Dict[str, Any] = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "adapter_dir": str(adapter_dir),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "head_to_head_samples": len(eval_records),
        "total_test_split": len(splits.test),
        "num_labels": len(splits.label2id),
        "models": {
            "qwen3_0.6b_base_zero_shot": base_zs,
            "qwen3_0.6b_base_few_shot": base_fs,
            "qwen3_0.6b_lora_finetuned": lora_res,
        },
        "delta_lora_vs_base_zero_shot": {
            "exact_match_pct_gain": round(lora_res["exact_match_pct"] - base_zs["exact_match_pct"], 2),
            "macro_f1_pct_gain": round(lora_res["macro_f1_pct"] - base_zs["macro_f1_pct"], 2),
            "top3_pct_gain": round(lora_res["top3_pct"] - base_zs["top3_pct"], 2),
            "invalid_label_pct_reduction": round(base_zs["invalid_label_pct"] - lora_res["invalid_label_pct"], 2),
        },
        "delta_lora_vs_base_few_shot": {
            "exact_match_pct_gain": round(lora_res["exact_match_pct"] - base_fs["exact_match_pct"], 2),
            "macro_f1_pct_gain": round(lora_res["macro_f1_pct"] - base_fs["macro_f1_pct"], 2),
            "top3_pct_gain": round(lora_res["top3_pct"] - base_fs["top3_pct"], 2),
            "invalid_label_pct_reduction": round(base_fs["invalid_label_pct"] - lora_res["invalid_label_pct"], 2),
        },
    }
    if lora_full_res is not None:
        results["lora_full_test_split"] = lora_full_res

    out_path = reports_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Eval] Saved results -> {out_path}", flush=True)

    del lora_model, base_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Head-to-head evaluation on Banking77 test split")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument(
        "--adapter-dir",
        default=str(PROJECT_DIR / "artifacts" / "qwen3_0.6b_banking77_lora"),
    )
    p.add_argument("--num-samples", type=int, default=770, help="Stratified test samples (0 = full 3076)")
    p.add_argument("--full-lora", action="store_true", help="Also run LoRA on full 3,076 test split")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--latency-samples", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_head_to_head(parse_args())
