"""Hybrid retrieval engine: FAISS IndexFlatIP + Okapi BM25 + Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import faiss
import numpy as np

from rag.embeddings import BaseEmbedder, LocalHybridEmbedder, l2_normalize, tokenize
from rag.models import DocumentChunk, SearchResult


class BM25Index:
    """Lightweight Okapi BM25 lexical ranker."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0
        self.num_docs: int = 0

    def fit(self, texts: Sequence[str]) -> BM25Index:
        self.doc_freqs = Counter()
        self.term_freqs = []
        self.doc_lengths = []
        self.num_docs = len(texts)

        total_len = 0
        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_doc_len = (total_len / self.num_docs) if self.num_docs > 0 else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        if df == 0 or self.num_docs == 0:
            return 0.0
        # BM25+ smoothed IDF ensures positive scores even in small corpora
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score_all(self, query: str) -> np.ndarray:
        if self.num_docs == 0:
            return np.zeros((0,), dtype=np.float32)

        q_tokens = tokenize(query)
        scores = np.zeros((self.num_docs,), dtype=np.float32)
        if not q_tokens or self.avg_doc_len <= 0:
            return scores

        for term in set(q_tokens):
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for idx, tf_map in enumerate(self.term_freqs):
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom

        return scores


class HybridVectorStore:
    """Hybrid FAISS (IndexFlatIP) + BM25 vector store with RRF fusion and disk persistence."""

    def __init__(self, embedder: BaseEmbedder | None = None) -> None:
        self.embedder: BaseEmbedder = embedder or LocalHybridEmbedder()
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []
        self.bm25: BM25Index = BM25Index()

    @property
    def size(self) -> int:
        return len(self.chunks)

    def build(self, chunks: Sequence[DocumentChunk]) -> int:
        """Fit embeddings, populate FAISS IndexFlatIP, and build BM25 index."""
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]

        if not self.chunks:
            self.index = faiss.IndexFlatIP(self.embedder.dim)
            self.bm25 = BM25Index().fit([])
            return 0

        vectors = self.embedder.fit_transform(texts)
        vectors = l2_normalize(np.ascontiguousarray(vectors, dtype=np.float32))
        self.index = faiss.IndexFlatIP(int(vectors.shape[1]))
        self.index.add(vectors)
        self.bm25 = BM25Index().fit(texts)
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[SearchResult]:
        """Search indexed chunks using hybrid RRF (FAISS + BM25), dense-only, or BM25-only."""
        if not self.chunks or self.index is None or top_k <= 0 or not query.strip():
            return []

        n = len(self.chunks)
        k_limit = min(top_k, n)
        candidate_k = min(n, max(k_limit * 4, 20))

        # 1. Dense FAISS IndexFlatIP search (cosine similarity on L2-normalized vectors)
        q_vec = self.embedder.embed_queries([query])
        q_vec = l2_normalize(np.ascontiguousarray(q_vec, dtype=np.float32))
        dense_scores_arr, dense_indices_arr = self.index.search(q_vec, candidate_k)

        dense_score_map: dict[int, float] = {}
        dense_rank_map: dict[int, int] = {}
        for rank_idx, (doc_idx, score) in enumerate(
            zip(dense_indices_arr[0].tolist(), dense_scores_arr[0].tolist()), start=1
        ):
            if doc_idx < 0:
                continue
            dense_score_map[doc_idx] = float(score)
            dense_rank_map[doc_idx] = rank_idx

        # 2. Sparse BM25 scoring
        bm25_all = self.bm25.score_all(query)
        bm25_order = np.argsort(-bm25_all).tolist()
        bm25_rank_map: dict[int, int] = {}
        rank_counter = 1
        for doc_idx in bm25_order[:candidate_k]:
            if float(bm25_all[doc_idx]) > 0.0:
                bm25_rank_map[doc_idx] = rank_counter
                rank_counter += 1

        # 3. Combine via Reciprocal Rank Fusion (RRF)
        candidate_ids = set(dense_rank_map.keys()) | set(bm25_rank_map.keys())
        scored_rows: list[tuple[int, float, float, float, float]] = []

        for doc_idx in candidate_ids:
            d_score = dense_score_map.get(doc_idx, 0.0)
            b_score = float(bm25_all[doc_idx])
            rrf_score = 0.0
            if doc_idx in dense_rank_map:
                rrf_score += dense_weight / (rrf_k + dense_rank_map[doc_idx])
            if doc_idx in bm25_rank_map:
                rrf_score += bm25_weight / (rrf_k + bm25_rank_map[doc_idx])

            if mode == "dense":
                final_score = d_score
            elif mode == "bm25":
                final_score = b_score
            elif mode == "hybrid":
                final_score = rrf_score
            else:
                raise ValueError(f"Unsupported search mode '{mode}'")

            scored_rows.append((doc_idx, final_score, d_score, b_score, rrf_score))

        # Deterministic tie-breaking: primary score desc, dense desc, bm25 desc, chunk_id asc
        scored_rows.sort(
            key=lambda row: (-row[1], -row[2], -row[3], self.chunks[row[0]].chunk_id)
        )

        results: list[SearchResult] = []
        for rank, (doc_idx, final_score, d_score, b_score, rrf_score) in enumerate(
            scored_rows[:k_limit], start=1
        ):
            results.append(
                SearchResult(
                    chunk=self.chunks[doc_idx],
                    score=final_score,
                    dense_score=d_score,
                    bm25_score=b_score,
                    rrf_score=rrf_score,
                    rank=rank,
                )
            )
        return results

    def save(self, index_dir: str | Path) -> tuple[Path, Path]:
        """Persist FAISS binary index (index.faiss) and chunk/embedder metadata (chunks.json)."""
        if self.index is None:
            raise RuntimeError("Cannot save an unbuilt vector store. Call build() first.")

        target_dir = Path(index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = target_dir / "index.faiss"
        chunks_path = target_dir / "chunks.json"

        faiss.write_index(self.index, str(faiss_path))
        payload = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_chunks": len(self.chunks),
            "dim": int(self.index.d),
            "backend": self.embedder.backend_name,
            "embedder_state": self.embedder.to_state(),
            "chunks": [c.to_dict() for c in self.chunks],
        }
        chunks_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return faiss_path, chunks_path

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        embedder_override: BaseEmbedder | None = None,
    ) -> HybridVectorStore:
        """Reload FAISS index.faiss + chunks.json and restore BM25 & embedder state."""
        target_dir = Path(index_dir)
        faiss_path = target_dir / "index.faiss"
        chunks_path = target_dir / "chunks.json"

        if not faiss_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Persisted index missing in {target_dir} (expected index.faiss and chunks.json)"
            )

        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        embedder = embedder_override or BaseEmbedder.from_state(payload["embedder_state"])

        store = cls(embedder=embedder)
        store.index = faiss.read_index(str(faiss_path))
        store.chunks = [DocumentChunk.from_dict(item) for item in payload["chunks"]]
        store.bm25 = BM25Index().fit([c.text for c in store.chunks])
        return store
