"""Unit tests for Project #2: Qwen3-0.6B LoRA fine-tuning & evaluation on Banking77."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import (  # noqa: E402
    DEFAULT_MODEL_ID,
    IGNORE_INDEX,
    DynamicSFTCollator,
    LengthBucketBatchSampler,
    build_few_shot_prompt,
    build_prompt,
    encode_training_example,
    load_banking77_splits,
    load_label_map,
    parse_predicted_label,
    save_label_map,
)
from src.eval import compute_classification_metrics, select_eval_subset  # noqa: E402
from src.infer import scores_to_confidences  # noqa: E402
from src.train_lora import compute_answer_only_loss  # noqa: E402


@pytest.fixture(scope="module")
def qwen_tokenizer():
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def banking77_splits():
    return load_banking77_splits(val_ratio=0.10, seed=42)


def test_banking77_has_all_77_canonical_labels(banking77_splits):
    assert len(banking77_splits.id2label) == 77
    assert len(banking77_splits.label2id) == 77
    assert set(banking77_splits.id2label.keys()) == set(range(77))
    assert len(banking77_splits.train) + len(banking77_splits.val) >= 9990
    assert len(banking77_splits.test) >= 3070
    # Every test label must exist in canonical 77 mapping
    test_labels = {r["label_text"] for r in banking77_splits.test}
    assert test_labels == set(banking77_splits.label2id.keys())


def test_stratified_val_split_preserves_all_77_classes(banking77_splits):
    train_classes = {r["label"] for r in banking77_splits.train}
    val_classes = {r["label"] for r in banking77_splits.val}
    assert len(train_classes) == 77
    assert len(val_classes) == 77


def test_prompt_masking_strictly_ignores_all_prompt_tokens(qwen_tokenizer):
    """Every prompt token MUST be masked with -100; ONLY answer tokens + EOS are supervised."""
    query = "My card was swallowed by the ATM yesterday!"
    label = "card_swallowed"
    item = encode_training_example(query, label, tokenizer=qwen_tokenizer, max_length=128)

    prompt_ids = qwen_tokenizer.encode(build_prompt(query), add_special_tokens=False)
    answer_ids = qwen_tokenizer.encode(label, add_special_tokens=False)
    expected_target = answer_ids + [qwen_tokenizer.eos_token_id]

    assert item["input_ids"] == prompt_ids + expected_target
    assert len(item["labels"]) == len(item["input_ids"])
    assert all(lbl == IGNORE_INDEX for lbl in item["labels"][: len(prompt_ids)])
    assert item["labels"][len(prompt_ids) :] == expected_target
    # Ensure non-masked tokens decode cleanly back to exact label
    supervised = [t for t in item["labels"] if t != IGNORE_INDEX]
    assert supervised[-1] == qwen_tokenizer.eos_token_id
    assert qwen_tokenizer.decode(supervised[:-1]) == label


def test_bpe_boundary_clean_for_all_77_labels(qwen_tokenizer, banking77_splits):
    """Concatenating prompt + label must never merge boundary tokens for any of the 77 intents."""
    prompt = build_prompt("I need help with my bank account.")
    p_ids = qwen_tokenizer.encode(prompt, add_special_tokens=False)
    for lbl in banking77_splits.id2label.values():
        a_ids = qwen_tokenizer.encode(lbl, add_special_tokens=False)
        full_ids = qwen_tokenizer.encode(prompt + lbl, add_special_tokens=False)
        assert full_ids == p_ids + a_ids, f"BPE boundary merge on label: {lbl}"


def test_truncation_preserves_answer_and_eos(qwen_tokenizer):
    """Even on overly long queries, truncation only trims prompt and never cuts answer + EOS."""
    long_query = "hello " * 200
    label = "wrong_exchange_rate_for_cash_withdrawal"
    item = encode_training_example(long_query, label, tokenizer=qwen_tokenizer, max_length=64)

    answer_ids = qwen_tokenizer.encode(label, add_special_tokens=False) + [qwen_tokenizer.eos_token_id]
    assert len(item["input_ids"]) == 64
    assert item["labels"][-len(answer_ids) :] == answer_ids
    assert all(lbl == IGNORE_INDEX for lbl in item["labels"][: -len(answer_ids)])


def test_dynamic_collator_pads_labels_with_ignore_index(qwen_tokenizer):
    collator = DynamicSFTCollator(pad_token_id=qwen_tokenizer.pad_token_id)
    ex1 = encode_training_example("Lost card", "lost_or_stolen_card", qwen_tokenizer)
    ex2 = encode_training_example(
        "Why did my international transfer get reverted back into my account?",
        "reverted_card_payment?",
        qwen_tokenizer,
    )
    batch = collator([ex1, ex2])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[1] % 8 == 0
    # Padded positions must have attention_mask==0 and label==-100
    pad_mask = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][pad_mask] == IGNORE_INDEX)


def test_lora_trainable_parameter_ratio_under_1_percent():
    """Tiny synthetic Qwen3 architecture with LoRA (r=8, q_proj/v_proj) keeps trainable ratio < 1%."""
    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).half()
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    ratio = trainable / total
    assert trainable > 0
    assert ratio < 0.01, f"Expected trainable ratio < 1%, got {ratio*100:.3f}%"


def test_parse_predicted_label_handles_clean_noisy_and_think_tags(banking77_splits):
    valid = list(banking77_splits.label2id.keys())
    assert parse_predicted_label("card_swallowed", valid) == "card_swallowed"
    assert parse_predicted_label("  `card_arrival`.\nSome explanation", valid) == "card_arrival"
    assert parse_predicted_label("Intent: change_pin\nDone", valid) == "change_pin"
    assert (
        parse_predicted_label("<think>Let me analyze the query...</think>\ncard_swallowed", valid)
        == "card_swallowed"
    )
    assert parse_predicted_label("card arrival", valid) == "card_arrival"
    assert parse_predicted_label("<think>unfinished reasoning", valid) == ""
    assert parse_predicted_label("completely_invalid_intent", valid) == "completely_invalid_intent"


def test_few_shot_prompt_contains_query_and_examples():
    p = build_few_shot_prompt("I got charged twice for coffee")
    assert "card_swallowed" in p
    assert "Query: I got charged twice for coffee" in p
    assert p.endswith("Intent:\n")


def test_classification_metrics_penalize_invalid_labels_and_track_top3():
    label2id = {f"intent_{i}": i for i in range(77)}
    gold = ["intent_0", "intent_1", "intent_2", "intent_3"]
    top1 = ["intent_0", "intent_9", "bogus_label", "intent_3"]
    top3 = [
        ["intent_0", "intent_1", "intent_2"],
        ["intent_9", "intent_1", "intent_4"],  # gold in top-3!
        ["bogus_label", "intent_5", "intent_6"],
        ["intent_3", "intent_0", "intent_1"],
    ]
    m = compute_classification_metrics(gold, top1, top3, label2id)
    assert math.isclose(m["exact_match_accuracy"], 0.5, rel_tol=1e-6)
    assert math.isclose(m["top3_accuracy"], 0.75, rel_tol=1e-6)
    assert math.isclose(m["invalid_label_rate"], 0.25, rel_tol=1e-6)
    assert m["invalid_label_count"] == 1
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_scores_to_confidences_softmax_properties():
    probs = scores_to_confidences([-0.1, -1.5, -3.2])
    assert len(probs) == 3
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
    assert probs[0] > probs[1] > probs[2] > 0.0


def test_label_map_save_and_load_roundtrip(tmp_path: Path, banking77_splits):
    out = tmp_path / "label_map.json"
    save_label_map(banking77_splits.id2label, out)
    id2l, l2id = load_label_map(out)
    assert id2l == banking77_splits.id2label
    assert l2id == banking77_splits.label2id


def test_select_eval_subset_stratified_covers_all_77_classes(banking77_splits):
    sub = select_eval_subset(banking77_splits.test, num_samples=770, seed=42)
    assert len(sub) == 770
    assert len({r["label"] for r in sub}) == 77


def test_length_bucket_batch_sampler_visits_every_index_once():
    lengths = [10, 80, 12, 75, 15, 90, 11, 82, 14, 70]
    sampler = LengthBucketBatchSampler(lengths, batch_size=3, mega_bucket_mult=4, seed=42)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))
    assert len(batches) == len(sampler)


def test_answer_only_loss_matches_full_causal_lm_loss():
    """Verify compute_answer_only_loss matches full HF CausalLM loss to < 1e-5."""
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=True,
    )
    base = Qwen3ForCausalLM(cfg)
    peft_model = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    peft_model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(5, 250, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, -4:] = input_ids[:, -4:]

    with torch.no_grad():
        std_loss = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        fast_loss, n_valid = compute_answer_only_loss(peft_model, input_ids, attention_mask, labels)

    assert n_valid == 8
    assert abs(float(std_loss.item()) - float(fast_loss.item())) < 1e-5
