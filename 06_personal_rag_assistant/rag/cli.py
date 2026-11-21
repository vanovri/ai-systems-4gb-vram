"""Command-line interface for Personal RAG Assistant (ingest, ask, eval, serve)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import uvicorn

from rag.api import create_app
from rag.embeddings import create_embedder
from rag.eval import evaluate_rag
from rag.ingest import ingest_directory, ingest_file
from rag.qa import GroundedQAEngine
from rag.store import HybridVectorStore


def cmd_ingest(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if source.is_dir():
        chunks = ingest_directory(
            source, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
    else:
        chunks = ingest_file(
            source, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

    if not chunks:
        print("Error: No chunks extracted from source.", file=sys.stderr)
        return 1

    embedder = create_embedder(backend=args.backend)
    store = HybridVectorStore(embedder=embedder)
    store.build(chunks)
    faiss_path, chunks_path = store.save(args.index_dir)

    summary = {
        "status": "indexed",
        "source": str(source),
        "num_sources": len({c.source_file for c in chunks}),
        "num_chunks": store.size,
        "backend": store.embedder.backend_name,
        "vector_dim": int(store.index.d) if store.index is not None else 0,
        "index_faiss": str(faiss_path),
        "chunks_json": str(chunks_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    store = HybridVectorStore.load(args.index_dir)
    engine = GroundedQAEngine(store=store, llm_mode=args.llm_mode)
    resp = engine.ask(args.query, top_k=args.top_k, search_mode=args.mode)

    if args.json:
        print(json.dumps(resp.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"Query:      {resp.query}")
        print(f"Refused:    {resp.refused}")
        print(f"Confidence: {resp.confidence:.4f} (mode={resp.mode})")
        print(f"Answer:\n  {resp.answer}")
        if resp.citations:
            print("Citations:")
            for c in resp.citations:
                print(f"  - {c.tag} (page={c.page}, score={c.relevance_score:.4f})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    index_path = Path(args.index_dir)
    if args.rebuild or not (index_path / "index.faiss").exists():
        chunks = ingest_directory(
            args.corpus, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        embedder = create_embedder(backend=args.backend)
        store = HybridVectorStore(embedder=embedder)
        store.build(chunks)
        store.save(index_path)
    else:
        store = HybridVectorStore.load(index_path)

    cases = None
    if args.dataset:
        cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    report = evaluate_rag(
        store=store,
        cases=cases,
        top_k=args.top_k,
        output_path=args.output,
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved full evaluation report to: {args.output}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    fastapi_app = create_app(index_dir=args.index_dir)
    uvicorn.run(fastapi_app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Personal RAG Assistant CLI (FAISS + BM25 + RRF + Grounded Citations)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest .txt/.md/.pdf files and build FAISS index")
    p_ingest.add_argument("source", help="Input file or corpus directory")
    p_ingest.add_argument("--index-dir", default="data/index", help="Output index directory")
    p_ingest.add_argument("--chunk-size", type=int, default=480)
    p_ingest.add_argument("--chunk-overlap", type=int, default=80)
    p_ingest.add_argument("--backend", choices=["auto", "local", "openai"], default="auto")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Query the indexed knowledge base with citations")
    p_ask.add_argument("query", help="Question to answer")
    p_ask.add_argument("--index-dir", default="data/index")
    p_ask.add_argument("--top-k", type=int, default=4)
    p_ask.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p_ask.add_argument("--llm-mode", choices=["auto", "extractive", "openai"], default="auto")
    p_ask.add_argument("--json", action="store_true", help="Output structured JSON")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="Evaluate Recall@3, MRR, and refusal accuracy")
    p_eval.add_argument("--corpus", default="data/corpus")
    p_eval.add_argument("--index-dir", default="data/index")
    p_eval.add_argument("--dataset", default=None, help="Optional JSON eval cases file")
    p_eval.add_argument("--output", default="reports/rag_eval.json")
    p_eval.add_argument("--top-k", type=int, default=3)
    p_eval.add_argument("--chunk-size", type=int, default=480)
    p_eval.add_argument("--chunk-overlap", type=int, default=80)
    p_eval.add_argument("--backend", choices=["auto", "local", "openai"], default="local")
    p_eval.add_argument("--rebuild", action="store_true", help="Rebuild index before evaluation")
    p_eval.set_defaults(func=cmd_eval)

    p_serve = sub.add_parser("serve", help="Run FastAPI server (/health, /ingest, /ask)")
    p_serve.add_argument("--index-dir", default="data/index")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
