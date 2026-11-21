"""Runnable CLI for Qwen3-0.6B + LoRA Banking77 intent classification.

Examples:
    python src/infer.py --text "My card was swallowed by the ATM"
    python src/infer.py --text "Why was I charged twice?" --top-k 3 --json
    python src/infer.py --file queries.txt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import (  # noqa: E402
    DEFAULT_MODEL_ID,
    build_prompt,
    load_label_map,
    parse_predicted_label,
)


def scores_to_confidences(seq_scores: Sequence[float]) -> List[float]:
    """Convert beam sequence log-scores into normalized probabilities summing to 1.0."""
    if not seq_scores:
        return []
    m = max(seq_scores)
    exps = [math.exp(float(s) - m) for s in seq_scores]
    denom = sum(exps)
    return [e / denom for e in exps] if denom > 0 else [1.0 / len(seq_scores)] * len(seq_scores)


@torch.no_grad()
def predict_intent(
    model: Any,
    tokenizer: Any,
    text: str,
    valid_labels: Optional[Sequence[str]] = None,
    top_k: int = 3,
    max_new_tokens: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Predict Banking77 intent + relative beam confidence + latency for a single query."""
    prompt = build_prompt(text)
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=128).to(device)
    prompt_len = enc["input_ids"].shape[1]

    num_beams = max(1, top_k)
    gen_cfg = GenerationConfig(
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
    )

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = model.generate(
            **enc,
            generation_config=gen_cfg,
            return_dict_in_generate=True,
            output_scores=True,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    gen_tokens = out.sequences[:, prompt_len:]
    raw_cands = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    if getattr(out, "sequences_scores", None) is not None:
        raw_scores = [float(x) for x in out.sequences_scores.detach().cpu().tolist()]
        probs = scores_to_confidences(raw_scores)
        top1_seq_prob = math.exp(min(raw_scores[0], 0.0))
    else:
        raw_scores = [0.0] * len(raw_cands)
        probs = [1.0 / max(len(raw_cands), 1)] * len(raw_cands)
        top1_seq_prob = 1.0

    candidates: List[Dict[str, Any]] = []
    valid_set = set(valid_labels) if valid_labels else None
    for raw_c, prob, logp in zip(raw_cands, probs, raw_scores):
        parsed = parse_predicted_label(raw_c, valid_labels=valid_labels)
        candidates.append(
            {
                "intent": parsed,
                "is_valid_banking77": (parsed in valid_set) if valid_set is not None else None,
                "relative_confidence": round(prob, 4),
                "seq_log_score": round(logp, 4),
                "raw_completion": raw_c.strip(),
            }
        )

    top1 = candidates[0] if candidates else {"intent": "", "relative_confidence": 0.0}
    return {
        "query": text,
        "intent": top1["intent"],
        "is_valid_banking77": top1.get("is_valid_banking77"),
        "confidence": round(top1_seq_prob, 4),
        "relative_beam_confidence": top1["relative_confidence"],
        "top_k": candidates,
        "latency_ms": round(latency_ms, 2),
    }


def load_inference_model(
    model_id: str,
    adapter_dir: Optional[Path],
    use_base: bool,
    device: str,
) -> tuple[Any, Any, Optional[List[str]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)

    valid_labels: Optional[List[str]] = None
    if adapter_dir is not None and (adapter_dir / "label_map.json").exists():
        _, label2id = load_label_map(adapter_dir / "label_map.json")
        valid_labels = sorted(label2id.keys())

    if not use_base:
        if adapter_dir is None or not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_dir}. Run src/train_lora.py first or pass --base."
            )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()

    model.eval()
    model.config.use_cache = True
    return model, tokenizer, valid_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Banking77 Intent Classification CLI (Qwen3-0.6B + LoRA)")
    p.add_argument("--text", type=str, default=None, help="Single banking customer query")
    p.add_argument("--file", type=str, default=None, help="Path to text file (one query per line)")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--adapter-dir",
        default=str(PROJECT_DIR / "artifacts" / "qwen3_0.6b_banking77_lora"),
    )
    p.add_argument("--base", action="store_true", help="Run un-finetuned Qwen3-0.6B base model")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    queries: List[str] = []
    if args.text:
        queries.append(args.text)
    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        queries.extend([ln.strip() for ln in lines if ln.strip()])
    if not queries:
        queries = ["My card was swallowed by the ATM"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    model, tokenizer, valid_labels = load_inference_model(
        model_id=args.model_id,
        adapter_dir=adapter_dir,
        use_base=args.base,
        device=device,
    )

    results = [
        predict_intent(
            model=model,
            tokenizer=tokenizer,
            text=q,
            valid_labels=valid_labels,
            top_k=args.top_k,
            device=device,
        )
        for q in queries
    ]

    if args.as_json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        mode_tag = "BASE" if args.base else "LoRA"
        for r in results:
            top3_str = ", ".join(
                f"{c['intent']} ({c['relative_confidence']*100:.1f}%)" for c in r["top_k"]
            )
            print(f"[{mode_tag}] Query:      {r['query']}")
            print(f"       Intent:     {r['intent']} (valid={r['is_valid_banking77']})")
            print(f"       Confidence: {r['confidence']*100:.2f}% (beam rel: {r['relative_beam_confidence']*100:.1f}%)")
            print(f"       Top-{args.top_k}:      {top3_str}")
            print(f"       Latency:    {r['latency_ms']:.2f} ms")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
