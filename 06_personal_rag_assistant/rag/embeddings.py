"""Dense embedding backends for FAISS IndexFlatIP.

Supports:
1. LocalHybridEmbedder: 100% offline deterministic TF-IDF/SVD + signed character n-gram
   and word unigram/bigram hashing producing L2-normalized float32 vectors.
2. OpenAICompatibleEmbedder: Uses OpenAI (or compatible) /embeddings endpoint when
   OPENAI_API_KEY is configured via environment variables (zero hardcoded secrets).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Sequence

import httpx
import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer preserving hyphen/underscore technical tokens."""
    return TOKEN_RE.findall(text.lower())


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization to unit vectors in float32 for cosine via IndexFlatIP."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    return (arr / safe_norms).astype(np.float32)


def _signed_hash_index(feature: str, dim: int) -> tuple[int, float]:
    """Deterministic blake2b feature hashing returning (bucket, sign)."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "little", signed=False)
    bucket = val % dim
    sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
    return bucket, sign


class BaseEmbedder(ABC):
    """Interface shared by local and remote dense embedding models."""

    backend_name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of output vectors."""

    @abstractmethod
    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """Fit embedder (if stateful) and return L2-normalized float32 embeddings."""

    @abstractmethod
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed one or more query strings into L2-normalized float32 vectors."""

    @abstractmethod
    def to_state(self) -> dict[str, Any]:
        """Serialize embedder parameters for disk persistence."""

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BaseEmbedder:
        backend = state.get("backend", "local_tfidf_svd_ngram")
        if backend == "openai":
            return OpenAICompatibleEmbedder.from_state(state)
        return LocalHybridEmbedder.from_state(state)


class LocalHybridEmbedder(BaseEmbedder):
    """Deterministic offline dense embedder: TF-IDF + Truncated SVD + char/word n-gram hashing.

    Produces L2-normalized float32 vectors suitable for real FAISS IndexFlatIP search
    without requiring external API keys or GPU hardware.
    """

    backend_name = "local_tfidf_svd_ngram"

    def __init__(
        self,
        svd_dim: int = 96,
        hash_dim: int = 160,
        svd_weight: float = 0.55,
        hash_weight: float = 0.45,
        max_vocab: int = 4096,
    ) -> None:
        if svd_dim <= 0 or hash_dim <= 0:
            raise ValueError("svd_dim and hash_dim must be positive")
        self.svd_dim = svd_dim
        self.hash_dim = hash_dim
        self.svd_weight = svd_weight
        self.hash_weight = hash_weight
        self.max_vocab = max_vocab

        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros((0,), dtype=np.float32)
        self.components: np.ndarray = np.zeros((0, svd_dim), dtype=np.float32)
        self._fitted = False

    @property
    def dim(self) -> int:
        return self.svd_dim + self.hash_dim

    def _build_tfidf_matrix(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        n_docs = len(tokenized_docs)
        n_terms = len(self.vocab)
        if n_docs == 0 or n_terms == 0:
            return np.zeros((n_docs, n_terms), dtype=np.float32)

        mat = np.zeros((n_docs, n_terms), dtype=np.float32)
        for row, tokens in enumerate(tokenized_docs):
            counts = Counter(t for t in tokens if t in self.vocab)
            for term, tf in counts.items():
                col = self.vocab[term]
                mat[row, col] = (1.0 + math.log(tf)) * float(self.idf[col])
        return l2_normalize(mat)

    def _hash_encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.hash_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features: Counter[str] = Counter()

            # Word unigrams (strong weight) & bigrams
            for tok in tokens:
                features[f"w:{tok}"] += 3
                # Character 3..5-grams for morphological resilience
                if len(tok) >= 3:
                    padded = f"<{tok}>"
                    for n in (3, 4, 5):
                        if len(padded) >= n:
                            for i in range(len(padded) - n + 1):
                                features[f"c{n}:{padded[i:i+n]}"] += 1

            for i in range(len(tokens) - 1):
                features[f"b:{tokens[i]}_{tokens[i+1]}"] += 2

            for feat, freq in features.items():
                bucket, sign = _signed_hash_index(feat, self.hash_dim)
                weight = 1.0 + math.log(freq)
                out[row, bucket] += sign * weight

        return l2_normalize(out)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            self._fitted = True
            return np.zeros((0, self.dim), dtype=np.float32)

        tokenized = [tokenize(t) for t in text_list]
        df: Counter[str] = Counter()
        for tokens in tokenized:
            for tok in set(tokens):
                df[tok] += 1

        # Deterministic ordering: frequency descending, then alphabetical
        ranked_terms = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_vocab]
        self.vocab = {term: idx for idx, (term, _) in enumerate(ranked_terms)}

        n_docs = len(text_list)
        self.idf = np.array(
            [math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0 for term, _ in ranked_terms],
            dtype=np.float32,
        )

        tfidf = self._build_tfidf_matrix(tokenized)
        n_terms = len(self.vocab)
        self.components = np.zeros((n_terms, self.svd_dim), dtype=np.float32)

        if n_terms > 0 and n_docs > 0:
            # Economy SVD on float64 for numerical stability, then cast to float32
            _, _, vt = np.linalg.svd(tfidf.astype(np.float64), full_matrices=False)
            k = min(self.svd_dim, vt.shape[0])
            # Fix singular vector sign deterministically by largest absolute element
            vh_k = vt[:k, :].copy()
            for r in range(k):
                max_col = int(np.argmax(np.abs(vh_k[r])))
                if vh_k[r, max_col] < 0:
                    vh_k[r] *= -1.0
            self.components[:, :k] = vh_k.T.astype(np.float32)

        self._fitted = True
        return self._transform_with_tokenized(text_list, tokenized)

    def _transform_with_tokenized(
        self, texts: Sequence[str], tokenized: list[list[str]]
    ) -> np.ndarray:
        hash_vecs = self._hash_encode(texts)
        if len(self.vocab) > 0 and self.components.shape[0] == len(self.vocab):
            tfidf = self._build_tfidf_matrix(tokenized)
            svd_vecs = l2_normalize(tfidf @ self.components)
        else:
            svd_vecs = np.zeros((len(texts), self.svd_dim), dtype=np.float32)

        combined = np.concatenate(
            [svd_vecs * self.svd_weight, hash_vecs * self.hash_weight], axis=1
        )
        return l2_normalize(combined)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        tokenized = [tokenize(t) for t in text_list]
        return self._transform_with_tokenized(text_list, tokenized)

    def to_state(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "svd_dim": self.svd_dim,
            "hash_dim": self.hash_dim,
            "svd_weight": self.svd_weight,
            "hash_weight": self.hash_weight,
            "max_vocab": self.max_vocab,
            "vocab": self.vocab,
            "idf": [round(float(x), 6) for x in self.idf.tolist()],
            "components": [
                [round(float(v), 6) for v in row] for row in self.components.tolist()
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> LocalHybridEmbedder:
        embedder = cls(
            svd_dim=int(state["svd_dim"]),
            hash_dim=int(state["hash_dim"]),
            svd_weight=float(state.get("svd_weight", 0.55)),
            hash_weight=float(state.get("hash_weight", 0.45)),
            max_vocab=int(state.get("max_vocab", 4096)),
        )
        embedder.vocab = {str(k): int(v) for k, v in state.get("vocab", {}).items()}
        embedder.idf = np.asarray(state.get("idf", []), dtype=np.float32)
        comps = state.get("components", [])
        if comps:
            embedder.components = np.asarray(comps, dtype=np.float32)
        else:
            embedder.components = np.zeros((len(embedder.vocab), embedder.svd_dim), dtype=np.float32)
        embedder._fitted = True
        return embedder


class OpenAICompatibleEmbedder(BaseEmbedder):
    """OpenAI-compatible /embeddings client activated strictly via environment config."""

    backend_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
        dimension: int = 1536,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAICompatibleEmbedder")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
            "/"
        )
        self.timeout = timeout
        self._http_client = http_client
        self._dim = dimension

    @property
    def dim(self) -> int:
        return self._dim

    def _call_api(self, texts: Sequence[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self._dim), dtype=np.float32)

        payload = {"model": self.model, "input": text_list}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/embeddings"

        if self._http_client is not None:
            resp = self._http_client.post(url, json=payload, headers=headers)
        else:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload, headers=headers)

        resp.raise_for_status()
        body = resp.json()
        items = sorted(body["data"], key=lambda item: item["index"])
        matrix = np.asarray([item["embedding"] for item in items], dtype=np.float32)
        self._dim = int(matrix.shape[1])
        return l2_normalize(matrix)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self._call_api(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._call_api(texts)

    def to_state(self) -> dict[str, Any]:
        # Never persist secrets to disk!
        return {
            "backend": self.backend_name,
            "model": self.model,
            "base_url": self.base_url,
            "dim": self._dim,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> OpenAICompatibleEmbedder:
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=state.get("model"),
            base_url=state.get("base_url"),
            dimension=int(state.get("dim", 1536)),
        )


def create_embedder(
    backend: str = "auto",
    svd_dim: int = 96,
    hash_dim: int = 160,
) -> BaseEmbedder:
    """Select OpenAI embedder when explicitly requested or when OPENAI_API_KEY is set."""
    mode = backend.lower().strip()
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())

    if mode == "openai" or (mode == "auto" and has_key):
        return OpenAICompatibleEmbedder()
    if mode in {"auto", "local"}:
        return LocalHybridEmbedder(svd_dim=svd_dim, hash_dim=hash_dim)
    raise ValueError(f"Unknown embedder backend '{backend}'. Expected 'auto', 'local', or 'openai'.")
