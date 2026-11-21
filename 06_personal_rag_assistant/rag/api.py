"""FastAPI application exposing /health, /ingest, and /ask endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.embeddings import create_embedder
from rag.ingest import ingest_directory, ingest_file, ingest_raw_document
from rag.models import DocumentChunk
from rag.qa import GroundedQAEngine
from rag.store import HybridVectorStore


class RawDocumentInput(BaseModel):
    source_file: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    page: int = Field(default=1, ge=1)


class IngestRequest(BaseModel):
    path: str | None = Field(default=None, description="File or directory path to ingest")
    documents: list[RawDocumentInput] | None = Field(
        default=None, description="Inline raw documents to ingest"
    )
    chunk_size: int = Field(default=500, ge=50, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)
    backend: Literal["auto", "local", "openai"] = "auto"


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    search_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"


def create_app(index_dir: str | Path | None = None) -> FastAPI:
    """Create a configured FastAPI app backed by a persistent HybridVectorStore."""
    resolved_index_dir = Path(
        index_dir or os.getenv("RAG_INDEX_DIR", "data/index")
    )

    app = FastAPI(
        title="Personal RAG Assistant API",
        version="1.0.0",
        description="Hybrid FAISS + BM25 + RRF retrieval with strict inline citation grounding.",
    )
    app.state.index_dir = resolved_index_dir
    app.state.store = None

    # Auto-load persisted index if present
    if (resolved_index_dir / "index.faiss").exists() and (resolved_index_dir / "chunks.json").exists():
        app.state.store = HybridVectorStore.load(resolved_index_dir)

    @app.get("/health")
    def health() -> dict[str, object]:
        store: HybridVectorStore | None = app.state.store
        return {
            "status": "ok",
            "index_loaded": store is not None,
            "num_chunks": store.size if store is not None else 0,
            "backend": store.embedder.backend_name if store is not None else None,
            "index_dir": str(app.state.index_dir),
        }

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, object]:
        if req.chunk_overlap >= req.chunk_size:
            raise HTTPException(
                status_code=422, detail="chunk_overlap must be smaller than chunk_size"
            )
        if not req.path and not req.documents:
            raise HTTPException(
                status_code=422, detail="Provide either 'path' or 'documents' to ingest"
            )

        chunks: list[DocumentChunk] = []
        if req.path:
            target = Path(req.path)
            if not target.exists():
                raise HTTPException(status_code=404, detail=f"Path not found: {target}")
            if target.is_dir():
                chunks.extend(
                    ingest_directory(
                        target, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap
                    )
                )
            else:
                chunks.extend(
                    ingest_file(target, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
                )

        if req.documents:
            for doc in req.documents:
                chunks.extend(
                    ingest_raw_document(
                        text=doc.text,
                        source_file=doc.source_file,
                        page=doc.page,
                        chunk_size=req.chunk_size,
                        chunk_overlap=req.chunk_overlap,
                    )
                )

        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable text chunks found")

        embedder = create_embedder(backend=req.backend)
        store = HybridVectorStore(embedder=embedder)
        store.build(chunks)
        faiss_path, chunks_path = store.save(app.state.index_dir)
        app.state.store = store

        return {
            "status": "indexed",
            "num_chunks": store.size,
            "num_sources": len({c.source_file for c in chunks}),
            "backend": store.embedder.backend_name,
            "index_faiss": str(faiss_path),
            "chunks_json": str(chunks_path),
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, object]:
        store: HybridVectorStore | None = app.state.store
        if store is None:
            if (app.state.index_dir / "index.faiss").exists():
                store = HybridVectorStore.load(app.state.index_dir)
                app.state.store = store
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No index loaded. Call POST /ingest or run CLI ingest first.",
                )

        engine = GroundedQAEngine(store=store)
        response = engine.ask(
            query=req.query,
            top_k=req.top_k,
            search_mode=req.search_mode,
        )
        return response.to_dict()

    return app


app = create_app()
