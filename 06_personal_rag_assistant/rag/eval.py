"""Retrieval and grounding evaluation suite (Recall@3, Recall@1, MRR, Refusal Accuracy)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.qa import GroundedQAEngine
from rag.store import HybridVectorStore


def default_eval_cases() -> list[dict[str, Any]]:
    """Curated evaluation benchmark spanning all 5 corpus documents (MD, TXT, PDF) + refusals."""
    return [
        {
            "id": "q01_faiss_cosine",
            "query": "How does FAISS IndexFlatIP compute cosine similarity?",
            "expected_source": "01_faiss_architecture.md",
            "expected_keywords": ["l2", "normalize", "inner product"],
            "should_refuse": False,
        },
        {
            "id": "q02_faiss_persistence",
            "query": "Which FAISS functions persist and reload a binary index from disk?",
            "expected_source": "01_faiss_architecture.md",
            "expected_keywords": ["write_index", "read_index"],
            "should_refuse": False,
        },
        {
            "id": "q03_bm25_params",
            "query": "What do the k1 and b hyperparameters control in Okapi BM25?",
            "expected_source": "02_bm25_and_rrf.md",
            "expected_keywords": ["saturation", "length"],
            "should_refuse": False,
        },
        {
            "id": "q04_rrf_formula",
            "query": "What is the Reciprocal Rank Fusion formula and default constant k?",
            "expected_source": "02_bm25_and_rrf.md",
            "expected_keywords": ["60", "rank"],
            "should_refuse": False,
        },
        {
            "id": "q05_citation_format",
            "query": "Why must grounded RAG systems attach inline source#chunk_id citations?",
            "expected_source": "03_rag_grounding_and_citations.txt",
            "expected_keywords": ["audit", "provenance"],
            "should_refuse": False,
        },
        {
            "id": "q06_anti_hallucination",
            "query": "When should a RAG assistant emit an explicit anti-hallucination refusal?",
            "expected_source": "03_rag_grounding_and_citations.txt",
            "expected_keywords": ["refusal", "threshold"],
            "should_refuse": False,
        },
        {
            "id": "q07_fastapi_health",
            "query": "What fields should a production FastAPI health endpoint expose?",
            "expected_source": "04_fastapi_production_ops.md",
            "expected_keywords": ["health", "status"],
            "should_refuse": False,
        },
        {
            "id": "q08_fastapi_lifespan",
            "query": "How does FastAPI avoid blocking the asyncio event loop during CPU work?",
            "expected_source": "04_fastapi_production_ops.md",
            "expected_keywords": ["threadpool", "event loop"],
            "should_refuse": False,
        },
        {
            "id": "q09_pdf_event_loop",
            "query": "How does Python asyncio schedule coroutines on a single-threaded event loop?",
            "expected_source": "05_python_async_internals.pdf",
            "expected_keywords": ["await", "coroutine"],
            "should_refuse": False,
        },
        {
            "id": "q10_pdf_cancellation",
            "query": "How should asyncio CancelledError be handled during graceful shutdown?",
            "expected_source": "05_python_async_internals.pdf",
            "expected_keywords": ["cancellederror", "cleanup"],
            "should_refuse": False,
        },
        {
            "id": "q11_oos_recipe",
            "query": "What temperature bakes sourdough brioche croissants in Paris?",
            "expected_source": None,
            "expected_keywords": [],
            "should_refuse": True,
        },
        {
            "id": "q12_oos_sports",
            "query": "Which goalkeeper saved the penalty shootout in the 1994 FIFA World Cup final?",
            "expected_source": None,
            "expected_keywords": [],
            "should_refuse": True,
        },
    ]


def evaluate_rag(
    store: HybridVectorStore,
    cases: list[dict[str, Any]] | None = None,
    top_k: int = 3,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run retrieval & grounding evaluation and optionally save JSON report."""
    eval_cases = cases if cases is not None else default_eval_cases()
    qa_engine = GroundedQAEngine(store=store, llm_mode="extractive")

    answerable_total = 0
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks: list[float] = []
    citations_valid_count = 0

    refusal_total = 0
    refusal_correct = 0

    mode_comparison: dict[str, dict[str, float]] = {}

    for mode in ("hybrid", "dense", "bm25"):
        m_hits3 = 0
        m_rrs: list[float] = []
        m_total = 0
        for case in eval_cases:
            if case.get("should_refuse"):
                continue
            m_total += 1
            results = store.search(case["query"], top_k=top_k, mode=mode)
            expected_src = case["expected_source"]
            first_rank = None
            for idx, r in enumerate(results, start=1):
                if r.chunk.source_file == expected_src:
                    first_rank = idx
                    break
            if first_rank is not None:
                m_hits3 += 1
                m_rrs.append(1.0 / first_rank)
            else:
                m_rrs.append(0.0)
        mode_comparison[mode] = {
            "recall_at_3": round(m_hits3 / max(1, m_total), 4),
            "mrr": round(sum(m_rrs) / max(1, m_total), 4),
        }

    details: list[dict[str, Any]] = []
    for case in eval_cases:
        query = case["query"]
        should_refuse = bool(case.get("should_refuse", False))
        expected_src = case.get("expected_source")

        results = store.search(query, top_k=top_k, mode="hybrid")
        qa_resp = qa_engine.ask(query, top_k=top_k, search_mode="hybrid")

        if should_refuse:
            refusal_total += 1
            passed_refusal = qa_resp.refused is True and len(qa_resp.citations) == 0
            if passed_refusal:
                refusal_correct += 1
            details.append(
                {
                    "id": case["id"],
                    "query": query,
                    "should_refuse": True,
                    "refused": qa_resp.refused,
                    "confidence": qa_resp.confidence,
                    "passed": passed_refusal,
                }
            )
        else:
            answerable_total += 1
            first_match_rank: int | None = None
            for idx, r in enumerate(results, start=1):
                if r.chunk.source_file == expected_src:
                    first_match_rank = idx
                    break

            hit1 = first_match_rank == 1
            hit3 = first_match_rank is not None and first_match_rank <= top_k
            rr = (1.0 / first_match_rank) if first_match_rank is not None else 0.0

            if hit1:
                hits_at_1 += 1
            if hit3:
                hits_at_3 += 1
            reciprocal_ranks.append(rr)

            cited_sources = {c.source_file for c in qa_resp.citations}
            citation_ok = (not qa_resp.refused) and (expected_src in cited_sources)
            if citation_ok:
                citations_valid_count += 1

            details.append(
                {
                    "id": case["id"],
                    "query": query,
                    "expected_source": expected_src,
                    "first_match_rank": first_match_rank,
                    "reciprocal_rank": round(rr, 4),
                    "hit_at_1": hit1,
                    "hit_at_3": hit3,
                    "refused": qa_resp.refused,
                    "citation_ok": citation_ok,
                    "citations": [c.tag for c in qa_resp.citations],
                    "top_sources": [r.chunk.source_file for r in results],
                    "confidence": qa_resp.confidence,
                }
            )

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_stats": {
            "num_chunks": store.size,
            "vector_dim": int(store.index.d) if store.index is not None else 0,
            "embedder_backend": store.embedder.backend_name,
            "faiss_index_type": type(store.index).__name__ if store.index is not None else None,
        },
        "metrics": {
            "answerable_queries": answerable_total,
            "refusal_queries": refusal_total,
            "recall_at_1": round(hits_at_1 / max(1, answerable_total), 4),
            "recall_at_3": round(hits_at_3 / max(1, answerable_total), 4),
            "mrr": round(sum(reciprocal_ranks) / max(1, answerable_total), 4),
            "grounded_citation_accuracy": round(
                citations_valid_count / max(1, answerable_total), 4
            ),
            "anti_hallucination_refusal_accuracy": round(
                refusal_correct / max(1, refusal_total), 4
            ),
        },
        "retrieval_mode_comparison": mode_comparison,
        "cases": details,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report
