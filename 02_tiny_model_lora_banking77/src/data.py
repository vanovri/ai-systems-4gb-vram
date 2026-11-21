"""Dataset loading, deterministic prompt formatting, and strict -100 answer-only loss masking for Banking77."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Sampler

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_DATASET_ID = "mteb/banking77"
IGNORE_INDEX = -100

# Fixed 5-shot examples (taken from distinct canonical intents) used for fair Base few-shot evaluation.
FEW_SHOT_EXAMPLES: Tuple[Tuple[str, str], ...] = (
    ("I am still waiting on my card to arrive.", "card_arrival"),
    ("My card was swallowed by the ATM machine.", "card_swallowed"),
    ("How do I change the PIN on my card?", "change_pin"),
    ("I need to verify my identity for my account.", "verify_my_identity"),
    ("Why did my transfer fail to go through?", "failed_transfer"),
)


@dataclass
class Banking77Splits:
    """Clean train/val/test splits plus canonical 77-intent bidirectional mappings."""

    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    test: List[Dict[str, Any]]
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    dataset_source: str


def build_prompt(query: str) -> str:
    """Deterministic, concise non-thinking prompt with a clean BPE newline boundary before the intent label."""
    clean_query = " ".join(str(query).strip().split())
    return f"Classify the banking customer query into one intent.\nQuery: {clean_query}\nIntent:\n"


def build_few_shot_prompt(
    query: str,
    examples: Sequence[Tuple[str, str]] = FEW_SHOT_EXAMPLES,
) -> str:
    """Concise 5-shot prompt for fair Base zero/few-shot evaluation using snake_case Banking77 labels."""
    clean_query = " ".join(str(query).strip().split())
    lines = [
        "Classify the banking customer query into one snake_case intent label. Output only the intent label.",
        "",
    ]
    for ex_q, ex_lbl in examples:
        lines.append(f"Query: {ex_q}")
        lines.append(f"Intent:\n{ex_lbl}")
        lines.append("")
    lines.append(f"Query: {clean_query}")
    lines.append("Intent:\n")
    return "\n".join(lines)


def encode_training_example(
    query: str,
    label_text: str,
    tokenizer: Any,
    max_length: int = 128,
) -> Dict[str, List[int]]:
    """Tokenize (prompt + answer + EOS) with strict IGNORE_INDEX (-100) masking on all prompt tokens.

    Implements the exact answer-only cross-entropy objective:
        L = -sum_{t in answer} log p_theta(y_t | y_{<t}, x)

    Guarantees:
    - Every prompt token has label == -100.
    - Every answer token AND final EOS token has label == token_id.
    - If total length exceeds `max_length`, only the prompt prefix is truncated from the right
      (never dropping answer tokens or EOS).
    """
    if max_length < 16:
        raise ValueError(f"max_length too small ({max_length}); must be >= 16")

    prompt_str = build_prompt(query)
    prompt_ids: List[int] = list(tokenizer.encode(prompt_str, add_special_tokens=False))
    answer_ids: List[int] = list(tokenizer.encode(label_text.strip(), add_special_tokens=False))
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define `eos_token_id`")

    target_ids = answer_ids + [int(eos_id)]
    max_prompt_len = max_length - len(target_ids)
    if max_prompt_len <= 0:
        raise ValueError(
            f"Target label '{label_text}' ({len(target_ids)} tokens) exceeds max_length={max_length}"
        )
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[:max_prompt_len]

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def parse_predicted_label(raw_text: str, valid_labels: Optional[Sequence[str]] = None) -> str:
    """Extract canonical snake_case intent label from raw model completion.

    Strips optional <think>...</think> blocks if emitted by base Qwen3, takes the first
    non-empty line, strips markdown/quotes/punctuation, and normalizes to lowercase snake_case.
    """
    text = str(raw_text or "")
    # Remove any Qwen3 thinking trace if present (common on base zero-shot)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    if not text:
        return ""

    first_line = ""
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            first_line = candidate
            break

    # Strip leading "Intent:" echo if model repeated the prefix
    first_line = re.sub(r"^(intent|label)\s*[:\-]\s*", "", first_line, flags=re.IGNORECASE).strip()
    first_line = first_line.strip("`\"'.,;:!?()[]{} ")
    # Take first whitespace-delimited token if the model continued explaining
    token = first_line.split()[0] if first_line.split() else ""
    token = token.strip("`\"'.,;:!?()[]{} ").lower()

    if valid_labels is not None:
        valid_set = set(valid_labels)
        if token in valid_set:
            return token
        # Also allow space-separated form matching snake_case (e.g. "card arrival" -> "card_arrival")
        underscored = re.sub(r"[\s\-]+", "_", first_line.lower()).strip("_")
        if underscored in valid_set:
            return underscored
    return token


def load_banking77_splits(
    val_ratio: float = 0.10,
    seed: int = 42,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> Banking77Splits:
    """Load real Banking77 from HuggingFace with stratified train/val split and untouched test split."""
    ds = load_dataset(dataset_id)

    def _normalize_split(split_ds: Any) -> List[Dict[str, Any]]:
        has_label_text = "label_text" in split_ds.column_names
        label_feature = split_ds.features.get("label")
        label_names = getattr(label_feature, "names", None)
        rows: List[Dict[str, Any]] = []
        for row in split_ds:
            lbl_id = int(row["label"])
            if has_label_text:
                lbl_text = str(row["label_text"]).strip()
            elif label_names is not None:
                lbl_text = str(label_names[lbl_id]).strip()
            else:
                raise ValueError("Dataset lacks both `label_text` column and ClassLabel names")
            rows.append({"text": str(row["text"]), "label": lbl_id, "label_text": lbl_text})
        return rows

    full_train = _normalize_split(ds["train"])
    test_rows = _normalize_split(ds["test"])

    pairs = sorted({(r["label"], r["label_text"]) for r in full_train})
    id2label = {idx: name for idx, name in pairs}
    label2id = {name: idx for idx, name in pairs}
    if len(id2label) != 77:
        raise ValueError(f"Expected 77 unique Banking77 labels, found {len(id2label)}")

    if val_ratio > 0.0:
        y_all = [r["label"] for r in full_train]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(full_train, y_all))
        train_rows = [full_train[i] for i in train_idx]
        val_rows = [full_train[i] for i in val_idx]
    else:
        train_rows = full_train
        val_rows = []

    return Banking77Splits(
        train=train_rows,
        val=val_rows,
        test=test_rows,
        id2label=id2label,
        label2id=label2id,
        dataset_source=dataset_id,
    )


def save_label_map(id2label: Dict[int, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_labels": len(id2label),
        "id2label": {str(k): v for k, v in sorted(id2label.items())},
        "label2id": {v: k for k, v in sorted(id2label.items())},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_label_map(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    id2label = {int(k): str(v) for k, v in data["id2label"].items()}
    label2id = {str(k): int(v) for k, v in data["label2id"].items()}
    return id2label, label2id


class Banking77SFTDataset(Dataset):
    """Pre-tokenized PyTorch Dataset with strict -100 masking on prompt tokens."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.samples: List[Dict[str, List[int]]] = [
            encode_training_example(
                query=r["text"],
                label_text=r["label_text"],
                tokenizer=tokenizer,
                max_length=max_length,
            )
            for r in records
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]


class DynamicSFTCollator:
    """Pads each batch dynamically to its longest sequence (capped at max_length) with -100 label padding."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        # Pad to multiple of 8 for Tensor Core friendly shapes on Turing FP16
        max_len = ((max_len + 7) // 8) * 8

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []

        for item in batch:
            pad_n = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_n)
            attention_mask.append(item["attention_mask"] + [0] * pad_n)
            labels.append(item["labels"] + [IGNORE_INDEX] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Groups similar-length sequences into batches, then shuffles batch order every epoch.

    Cuts padding waste by ~2x on short Banking77 queries while preserving stochastic batch order.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        mega_bucket_mult: int = 32,
        seed: int = 42,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.mega_bucket_size = max(self.batch_size * mega_bucket_mult, self.batch_size)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        perm = torch.randperm(len(self.lengths), generator=rng).tolist()
        batches: List[List[int]] = []
        for start in range(0, len(perm), self.mega_bucket_size):
            chunk = perm[start : start + self.mega_bucket_size]
            chunk.sort(key=lambda idx: self.lengths[idx])
            for b_start in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b_start : b_start + self.batch_size])

        batch_order = torch.randperm(len(batches), generator=rng).tolist()
        for b_idx in batch_order:
            yield batches[b_idx]
