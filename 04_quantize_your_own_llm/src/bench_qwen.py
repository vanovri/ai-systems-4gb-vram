"""Benchmark pure-NumPy group-wise INT8/INT4 quantization on real Qwen/Qwen3-0.6B.

Measures strictly real numbers (zero fabricated metrics):
1. WikiText-2 test Perplexity (FP16 vs INT8 asym g=64 vs INT4 asym g=64 vs INT4 sym g=64)
2. Exact relative L1 weight error (epsilon), RMSE, and actual packed RAM/disk byte accounting
   (both quantized linear layers only AND full model with tied FP16 embeddings/lm_head)
3. Layer-10 real weight visualization (64 real weights vs 16 INT4 snap levels -> reports/layer10_int4_snap.png)
4. Layer-10 GPTQ-lite (Hessian-aware Cholesky) vs RTN output reconstruction error on real WikiText-2 activations
5. Cross-project synergy: Banking77 accuracy of LoRA-merged Qwen3-0.6B in FP16 vs quantized INT4 (g=64)
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_ORIG_LINEAR = F.linear


@contextlib.contextmanager
def tu117_fast_inference_ctx(device: str = "cuda"):
    """Accelerates FP16-stored Linear forward ops on Tensor-Core-less GTX 16xx (TU116/TU117)."""
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
REPO_ROOT = PROJECT_DIR.parent
PROJECT2_DIR = REPO_ROOT / "02_tiny_model_lora_banking77"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.quantizer import (  # noqa: E402
    QuantMode,
    compute_quant_metrics,
    dequantize_groupwise,
    gptq_lite_quantize,
    quantize_groupwise,
)
from src.visualize import plot_benchmark_summary, plot_layer10_int4_snap  # noqa: E402

QUANT_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def is_target_linear(name: str, module: nn.Module) -> bool:
    """Quantize transformer attention & MLP Linear projections; keep embeddings/lm_head/norms FP16."""
    if not isinstance(module, nn.Linear):
        return False
    if "lm_head" in name or "embed" in name:
        return False
    return any(name.endswith(sfx) for sfx in QUANT_LINEAR_SUFFIXES)


def compute_model_fp16_byte_breakdown(model: nn.Module) -> Dict[str, int]:
    """Deduplicate tied tensors (e.g. embed_tokens & lm_head) for exact FP16 byte accounting."""
    seen_ptrs = set()
    quant_target_params = 0
    non_quant_params = 0

    quant_param_ptrs = set()
    for name, mod in model.named_modules():
        if is_target_linear(name, mod):
            quant_param_ptrs.add(mod.weight.data_ptr())

    for _, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        if ptr in quant_param_ptrs:
            quant_target_params += param.numel()
        else:
            non_quant_params += param.numel()

    return {
        "quant_target_params": int(quant_target_params),
        "non_quant_params": int(non_quant_params),
        "total_unique_params": int(quant_target_params + non_quant_params),
        "quant_target_fp16_bytes": int(quant_target_params * 2),
        "non_quant_fp16_bytes": int(non_quant_params * 2),
        "total_fp16_bytes": int((quant_target_params + non_quant_params) * 2),
    }


def load_wikitext2_token_stream(tokenizer: Any, max_tokens: int = 32768) -> torch.Tensor:
    """Load real WikiText-2 test split and concatenate non-empty lines into a 1D token ID stream."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(line for line in ds["text"] if line and line.strip())
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if max_tokens > 0:
        token_ids = token_ids[:max_tokens]
    return torch.tensor(token_ids, dtype=torch.long)


@torch.no_grad()
def evaluate_wikitext2_ppl(
    model: nn.Module,
    token_stream: torch.Tensor,
    seq_len: int = 256,
    stride: int = 256,
    num_windows: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute exact sliding-window token-level Perplexity: PPL = exp(-1/N sum log p(x_t | x_{<t}))."""
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    n_stream = int(token_stream.numel())
    total_nll = 0.0
    total_eval_tokens = 0
    windows_evaluated = 0

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    prev_end = 0
    with tu117_fast_inference_ctx(device):
        for begin in range(0, n_stream - 1, stride):
            if num_windows > 0 and windows_evaluated >= num_windows:
                break
            end = min(begin + seq_len, n_stream)
            if end - begin < 2:
                break
            trg_len = end - prev_end
            if trg_len <= 0:
                break

            input_ids = token_stream[begin:end].unsqueeze(0).to(device)
            labels = input_ids.clone()
            # Mask overlapping context tokens so every evaluated token is counted exactly once
            context_len = labels.shape[1] - trg_len
            if context_len > 0:
                labels[:, :context_len] = -100

            # Number of actually predicted tokens inside causal LM shift
            eval_count = int((labels[:, 1:] != -100).sum().item())
            if eval_count <= 0:
                prev_end = end
                continue

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                outputs = model(input_ids=input_ids, labels=labels)
            mean_nll = float(outputs.loss.item())
            total_nll += mean_nll * eval_count
            total_eval_tokens += eval_count
            windows_evaluated += 1
            prev_end = end
            if end >= n_stream:
                break

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    avg_nll = total_nll / max(total_eval_tokens, 1)
    ppl = math.exp(avg_nll)
    return {
        "perplexity": round(ppl, 4),
        "mean_nll": round(avg_nll, 6),
        "evaluated_tokens": int(total_eval_tokens),
        "windows_evaluated": int(windows_evaluated),
        "seq_len": int(seq_len),
        "stride": int(stride),
        "eval_seconds": round(elapsed, 2),
    }


def quantize_model_in_place_numpy(
    model: nn.Module,
    bits: int,
    group_size: int = 64,
    mode: QuantMode = "asymmetric",
    save_orig_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Apply pure-NumPy group-wise quantization (with real INT4 nibble packing) to all target Linear layers.

    De-quantizes the REAL packed tensor (including FP16 scale/min metadata rounding) back to FP16
    for exact GPU forward-pass Perplexity measurement on GTX 1650 Ti.
    """
    pack = bits == 4
    total_fp16_bytes = 0
    total_quant_bytes = 0
    total_data_bytes = 0
    total_meta_bytes = 0
    total_l1_err = 0.0
    total_l1_orig = 0.0
    total_sq_err = 0.0
    total_numel = 0
    global_max_abs = 0.0
    layer_rel_l1s: List[float] = []
    num_layers_quantized = 0

    t0 = time.perf_counter()
    for name, mod in model.named_modules():
        if not is_target_linear(name, mod):
            continue
        if save_orig_weights is not None and name not in save_orig_weights:
            save_orig_weights[name] = mod.weight.detach().cpu().clone()

        w_src = (
            save_orig_weights[name]
            if (save_orig_weights is not None and name in save_orig_weights)
            else mod.weight.detach().cpu()
        )
        w_np = w_src.float().numpy()

        qt = quantize_groupwise(w_np, bits=bits, group_size=group_size, mode=mode, pack=pack)
        w_hat = dequantize_groupwise(qt, out_dtype=np.float32)

        # Accumulate exact weight errors and packed byte counts
        diff = w_hat.astype(np.float64) - w_np.astype(np.float64)
        l1_err = float(np.sum(np.abs(diff)))
        l1_orig = float(np.sum(np.abs(w_np.astype(np.float64))))
        total_l1_err += l1_err
        total_l1_orig += l1_orig
        total_sq_err += float(np.sum(diff * diff))
        total_numel += w_np.size
        global_max_abs = max(global_max_abs, float(np.max(np.abs(diff))))
        layer_rel_l1s.append(l1_err / max(l1_orig, 1e-8))

        total_fp16_bytes += qt.fp16_baseline_nbytes
        total_quant_bytes += qt.total_nbytes
        total_data_bytes += qt.data_nbytes
        total_meta_bytes += qt.meta_nbytes
        num_layers_quantized += 1

        # Copy dequantized weights back to model module on its current device
        mod.weight.data.copy_(torch.from_numpy(w_hat).to(dtype=mod.weight.dtype, device=mod.weight.device))

    elapsed = time.perf_counter() - t0
    breakdown = compute_model_fp16_byte_breakdown(model)
    full_quant_bytes = total_quant_bytes + breakdown["non_quant_fp16_bytes"]
    mib = 1024 ** 2

    return {
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
        "packed_nibbles": pack,
        "num_layers_quantized": num_layers_quantized,
        "quantized_params": total_numel,
        "quantize_seconds": round(elapsed, 2),
        "global_rel_l1_error": round(total_l1_err / max(total_l1_orig, 1e-8), 6),
        "global_rel_l1_error_pct": round(100.0 * total_l1_err / max(total_l1_orig, 1e-8), 4),
        "mean_rel_l1_error_pct": round(100.0 * float(np.mean(layer_rel_l1s)), 4),
        "global_rmse": round(math.sqrt(total_sq_err / max(total_numel, 1)), 6),
        "max_abs_error": round(global_max_abs, 6),
        "quantized_linear_fp16_mb": round(total_fp16_bytes / mib, 2),
        "quantized_linear_mb": round(total_quant_bytes / mib, 2),
        "quantized_linear_data_mb": round(total_data_bytes / mib, 2),
        "quantized_linear_meta_mb": round(total_meta_bytes / mib, 2),
        "effective_bits_per_quant_weight": round((total_quant_bytes * 8.0) / max(total_numel, 1), 3),
        "linear_compression_ratio_vs_fp16": round(total_fp16_bytes / max(total_quant_bytes, 1), 3),
        "full_model_fp16_mb": round(breakdown["total_fp16_bytes"] / mib, 2),
        "full_model_quant_mb": round(full_quant_bytes / mib, 2),
        "full_model_compression_ratio_vs_fp16": round(
            breakdown["total_fp16_bytes"] / max(full_quant_bytes, 1), 3
        ),
    }


def restore_weights(model: nn.Module, saved_weights: Dict[str, torch.Tensor]) -> None:
    for name, mod in model.named_modules():
        if name in saved_weights:
            mod.weight.data.copy_(saved_weights[name].to(dtype=mod.weight.dtype, device=mod.weight.device))


@torch.no_grad()
def run_layer10_gptq_lite_ablation(
    model: nn.Module,
    token_stream: torch.Tensor,
    device: str = "cuda",
    calib_windows: int = 8,
    seq_len: int = 256,
) -> Dict[str, Any]:
    """Capture real Layer-10 q_proj activations from WikiText-2 and compare GPTQ-lite vs RTN (INT4 g=64)."""
    target_mod = model.model.layers[10].self_attn.q_proj
    captured: List[np.ndarray] = []

    def _hook(_mod: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0].detach().float().cpu().numpy().reshape(-1, inputs[0].shape[-1])
        captured.append(x)

    handle = target_mod.register_forward_pre_hook(_hook)
    model.eval()
    try:
        for w_idx in range(calib_windows):
            begin = w_idx * seq_len
            end = min(begin + seq_len, int(token_stream.numel()))
            if end - begin < 2:
                break
            inp = token_stream[begin:end].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                _ = model(input_ids=inp)
    finally:
        handle.remove()

    x_calib = np.concatenate(captured, axis=0)
    w_layer10 = target_mod.weight.detach().float().cpu().numpy()

    t0 = time.perf_counter()
    _, gptq_stats = gptq_lite_quantize(
        w=w_layer10,
        x_calib=x_calib,
        bits=4,
        group_size=64,
        block_size=64,
        damp_percent=0.01,
    )
    elapsed = time.perf_counter() - t0
    return {
        "layer": "model.layers.10.self_attn.q_proj",
        "weight_shape": list(w_layer10.shape),
        "calibration_tokens": int(x_calib.shape[0]),
        "runtime_seconds": round(elapsed, 2),
        **{k: round(v, 6) for k, v in gptq_stats.items()},
    }


def run_cross_project_lora_int4_eval(
    model_id: str,
    adapter_dir: Path,
    num_samples: int,
    batch_size: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    """Cross-project synergy: evaluate merged Project #2 LoRA model in FP16 vs Pure-NumPy INT4 (g=64)."""
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"[Cross-Project] Adapter not found at {adapter_dir}; skipping.", flush=True)
        return None

    # Import Project #2 eval utilities cleanly via importlib to avoid `src` package collision
    import importlib.util

    def _load_mod(mod_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    p2_data = _load_mod("p2_data", PROJECT2_DIR / "src" / "data.py")
    # Temporarily bind Project #2 `src.data` only if needed while loading `eval.py`
    p2_eval_src = PROJECT2_DIR / "src" / "eval.py"
    # Cleanest import: load using sys.path priority for p2_eval
    orig_src_mod = sys.modules.get("src")
    orig_src_data = sys.modules.get("src.data")
    sys.modules["src.data"] = p2_data
    try:
        p2_eval = _load_mod("p2_eval", p2_eval_src)
    finally:
        if orig_src_data is not None:
            sys.modules["src.data"] = orig_src_data
        else:
            sys.modules.pop("src.data", None)
        if orig_src_mod is not None:
            sys.modules["src"] = orig_src_mod

    print(f"[Cross-Project] Loading Banking77 test subset (n={num_samples}) & LoRA adapter...", flush=True)
    splits = p2_data.load_banking77_splits(val_ratio=0.10, seed=42)
    eval_records = p2_eval.select_eval_subset(splits.test, num_samples=num_samples, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.eval()

    print("[Cross-Project] Evaluating LoRA-merged Qwen3-0.6B in FP16...", flush=True)
    fp16_metrics = p2_eval.evaluate_model_on_records(
        model=merged,
        tokenizer=tokenizer,
        records=eval_records,
        label2id=splits.label2id,
        prompt_mode="standard",
        batch_size=batch_size,
        max_new_tokens=16,
        num_beams=3,
        latency_samples=25,
        device=device,
    )

    print("[Cross-Project] Quantizing LoRA-merged Qwen3-0.6B to INT4 asymmetric (g=64, nibble-packed)...", flush=True)
    q_stats = quantize_model_in_place_numpy(merged, bits=4, group_size=64, mode="asymmetric")

    print("[Cross-Project] Evaluating LoRA-merged Qwen3-0.6B in INT4 (g=64)...", flush=True)
    int4_metrics = p2_eval.evaluate_model_on_records(
        model=merged,
        tokenizer=tokenizer,
        records=eval_records,
        label2id=splits.label2id,
        prompt_mode="standard",
        batch_size=batch_size,
        max_new_tokens=16,
        num_beams=3,
        latency_samples=25,
        device=device,
    )

    print(
        f"  LoRA FP16 -> INT4 (g=64): Exact-Match {fp16_metrics['exact_match_pct']}% -> "
        f"{int4_metrics['exact_match_pct']}% (delta={int4_metrics['exact_match_pct'] - fp16_metrics['exact_match_pct']:+.2f}%) | "
        f"Macro-F1 {fp16_metrics['macro_f1_pct']}% -> {int4_metrics['macro_f1_pct']}% | "
        f"Quantized Linear Size: {q_stats['quantized_linear_mb']} MB ({q_stats['linear_compression_ratio_vs_fp16']}x)",
        flush=True,
    )

    del merged, base
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "adapter_dir": str(adapter_dir),
        "num_test_samples": len(eval_records),
        "lora_fp16": {
            "exact_match_pct": fp16_metrics["exact_match_pct"],
            "macro_f1_pct": fp16_metrics["macro_f1_pct"],
            "top3_pct": fp16_metrics["top3_pct"],
            "invalid_label_pct": fp16_metrics["invalid_label_pct"],
            "latency_p50_ms": fp16_metrics["latency_p50_ms"],
        },
        "lora_int4_asym_g64": {
            "exact_match_pct": int4_metrics["exact_match_pct"],
            "macro_f1_pct": int4_metrics["macro_f1_pct"],
            "top3_pct": int4_metrics["top3_pct"],
            "invalid_label_pct": int4_metrics["invalid_label_pct"],
            "latency_p50_ms": int4_metrics["latency_p50_ms"],
            "quantization_stats": q_stats,
        },
        "accuracy_retention_delta_pct": round(
            int4_metrics["exact_match_pct"] - fp16_metrics["exact_match_pct"], 2
        ),
        "macro_f1_retention_delta_pct": round(
            int4_metrics["macro_f1_pct"] - fp16_metrics["macro_f1_pct"], 2
        ),
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Bench] Loading cached {args.model_id} on {device} (FP16)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(device)
    model.eval()

    breakdown = compute_model_fp16_byte_breakdown(model)
    mib = 1024 ** 2
    print(
        f"[Model] unique_params={breakdown['total_unique_params']:,} | "
        f"quant_target_linear={breakdown['quant_target_params']:,} ({breakdown['quant_target_fp16_bytes']/mib:.2f} MB FP16) | "
        f"non_quant_tied_embed_norms={breakdown['non_quant_params']:,} ({breakdown['non_quant_fp16_bytes']/mib:.2f} MB FP16)",
        flush=True,
    )

    # 1. Generate Layer-10 INT4 snap visualization from REAL weights before modifying model
    w64_layer10 = model.model.layers[10].self_attn.q_proj.weight[0, :64].detach().cpu().float().numpy()
    layer10_snap_info = plot_layer10_int4_snap(
        w64=w64_layer10,
        out_path=reports_dir / "layer10_int4_snap.png",
    )
    print(
        f"[Viz] Saved Layer-10 INT4 snap plot -> {reports_dir / 'layer10_int4_snap.png'} "
        f"(rel L1={layer10_snap_info['rel_l1_error']*100:.2f}%, {layer10_snap_info['packed_bytes']}B vs {layer10_snap_info['fp16_bytes']}B)",
        flush=True,
    )

    # 2. Load WikiText-2 test token stream
    print("[Data] Tokenizing WikiText-2 test split...", flush=True)
    token_stream = load_wikitext2_token_stream(tokenizer, max_tokens=args.num_windows * args.seq_len + 1024)
    print(f"[Data] Stream length={token_stream.numel():,} tokens | windows={args.num_windows} x {args.seq_len}", flush=True)

    # 3. Run Layer-10 GPTQ-lite Hessian-aware comparison vs RTN on real WikiText-2 activations
    print("[GPTQ-lite] Capturing real Layer-10 q_proj activations & comparing GPTQ-lite vs RTN...", flush=True)
    gptq_ablation = run_layer10_gptq_lite_ablation(
        model=model,
        token_stream=token_stream,
        device=device,
        calib_windows=8,
        seq_len=256,
    )
    print(
        f"  Layer-10 q_proj Output Rel L2 Err: RTN={gptq_ablation['rtn_output_rel_l2_error']*100:.3f}% vs "
        f"GPTQ-lite={gptq_ablation['gptq_output_rel_l2_error']*100:.3f}% "
        f"({gptq_ablation['output_error_reduction_pct']:+.2f}% error reduction in {gptq_ablation['runtime_seconds']}s)",
        flush=True,
    )

    # 4. Evaluate FP16 baseline Perplexity
    print("[PPL] Evaluating FP16 baseline on WikiText-2...", flush=True)
    fp16_ppl = evaluate_wikitext2_ppl(
        model, token_stream, seq_len=args.seq_len, stride=args.stride, num_windows=args.num_windows, device=device
    )
    fp16_entry = {
        "bits": 16,
        "group_size": None,
        "mode": "fp16",
        "packed_nibbles": False,
        "global_rel_l1_error": 0.0,
        "global_rel_l1_error_pct": 0.0,
        "mean_rel_l1_error_pct": 0.0,
        "global_rmse": 0.0,
        "max_abs_error": 0.0,
        "quantized_linear_fp16_mb": round(breakdown["quant_target_fp16_bytes"] / mib, 2),
        "quantized_linear_mb": round(breakdown["quant_target_fp16_bytes"] / mib, 2),
        "effective_bits_per_quant_weight": 16.0,
        "linear_compression_ratio_vs_fp16": 1.0,
        "full_model_fp16_mb": round(breakdown["total_fp16_bytes"] / mib, 2),
        "full_model_quant_mb": round(breakdown["total_fp16_bytes"] / mib, 2),
        "full_model_compression_ratio_vs_fp16": 1.0,
        **fp16_ppl,
        "delta_ppl": 0.0,
        "delta_ppl_pct": 0.0,
    }
    print(f"  FP16 Baseline:       PPL={fp16_ppl['perplexity']:.4f} ({fp16_ppl['evaluated_tokens']} tokens)", flush=True)

    # Save original FP16 weights in CPU RAM once so we cleanly test INT8, INT4-asym, INT4-sym without re-loading
    saved_orig_weights: Dict[str, torch.Tensor] = {}
    variants: Dict[str, Dict[str, Any]] = {"fp16": fp16_entry}

    configs: List[Tuple[str, int, QuantMode]] = [
        ("int8_asym_g64", 8, "asymmetric"),
        ("int4_asym_g64", 4, "asymmetric"),
        ("int4_sym_g64", 4, "symmetric"),
    ]
    for key, bits, mode in configs:
        print(f"[Quant] Quantizing 196 Linear layers -> {key} (bits={bits}, g={args.group_size}, mode={mode})...", flush=True)
        q_stats = quantize_model_in_place_numpy(
            model, bits=bits, group_size=args.group_size, mode=mode, save_orig_weights=saved_orig_weights
        )
        ppl_res = evaluate_wikitext2_ppl(
            model, token_stream, seq_len=args.seq_len, stride=args.stride, num_windows=args.num_windows, device=device
        )
        delta_ppl = ppl_res["perplexity"] - fp16_ppl["perplexity"]
        delta_pct = 100.0 * delta_ppl / max(fp16_ppl["perplexity"], 1e-8)
        entry = {
            **q_stats,
            **ppl_res,
            "delta_ppl": round(delta_ppl, 4),
            "delta_ppl_pct": round(delta_pct, 2),
        }
        variants[key] = entry
        print(
            f"  {key:15s}: PPL={ppl_res['perplexity']:.4f} (dPPL={delta_ppl:+.4f}, {delta_pct:+.2f}%) | "
            f"L1 eps={q_stats['global_rel_l1_error_pct']:.3f}% | "
            f"Linear={q_stats['quantized_linear_mb']:.1f} MB ({q_stats['linear_compression_ratio_vs_fp16']:.2f}x) | "
            f"Full={q_stats['full_model_quant_mb']:.1f} MB ({q_stats['full_model_compression_ratio_vs_fp16']:.2f}x)",
            flush=True,
        )

    # Free base model GPU memory before cross-project LoRA evaluation
    del model, saved_orig_weights
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 5. Cross-project synergy: LoRA Banking77 merged & quantized to INT4
    cross_project: Optional[Dict[str, Any]] = None
    if not args.skip_cross_project:
        adapter_dir = Path(args.lora_adapter_dir)
        cross_project = run_cross_project_lora_int4_eval(
            model_id=args.model_id,
            adapter_dir=adapter_dir,
            num_samples=args.cross_eval_samples,
            batch_size=16,
            device=device,
        )

    payload: Dict[str, Any] = {
        "model_id": args.model_id,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "group_size": args.group_size,
        "model_param_breakdown": breakdown,
        "wikitext2_benchmark": variants,
        "layer10_snap_visualization": layer10_snap_info,
        "layer10_gptq_lite_ablation": gptq_ablation,
        "cross_project_lora_banking77_int4": cross_project,
    }

    out_json = reports_dir / "quant_benchmark.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_benchmark_summary(payload, reports_dir / "quant_summary.png")
    print(f"[Bench] Saved benchmark JSON -> {out_json}", flush=True)
    print(f"[Bench] Saved summary chart  -> {reports_dir / 'quant_summary.png'}", flush=True)

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark pure-NumPy INT8/INT4 quantization on Qwen3-0.6B")
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--num-windows", type=int, default=64)
    p.add_argument(
        "--lora-adapter-dir",
        default=str(PROJECT2_DIR / "artifacts" / "qwen3_0.6b_banking77_lora"),
    )
    p.add_argument("--cross-eval-samples", type=int, default=385, help="Stratified Banking77 samples (5 per intent)")
    p.add_argument("--skip-cross-project", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
