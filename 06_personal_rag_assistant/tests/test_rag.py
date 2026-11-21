"""Comprehensive offline test suite for Project #6 — Personal RAG Assistant."""

from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.cli import main as cli_main
from rag.embeddings import LocalHybridEmbedder, OpenAICompatibleEmbedder, create_embedder
from rag.eval import evaluate_rag
from rag.ingest import chunk_text, ingest_directory, ingest_file, ingest_raw_document
from rag.qa import REFUSAL_MESSAGE, GroundedQAEngine
from rag.store import BM25Index, HybridVectorStore
from scripts.generate_sample_pdf import build_text_pdf

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


def test_chunk_text_overlap_and_validation() -> None:
    text = (
        "First sentence explains FAISS exact search. "
        "Second sentence describes L2 normalization for cosine similarity. "
        "Third sentence covers BM25 lexical scoring with k1 and b. "
        "Fourth sentence summarizes Reciprocal Rank Fusion."
    )
    chunks = chunk_text(text, chunk_size=110, chunk_overlap=25)
    assert len(chunks) >= 2
    for chunk_body, start_char, end_char in chunks:
        assert len(chunk_body) <= 110
        assert start_char >= 0
        assert end_char >= start_char

    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=50, chunk_overlap=50)


def test_ingest_pdf_preserves_multipage_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample_notes.pdf"
    build_text_pdf(
        pages=[
            ["Page one covers Kubernetes readiness probes and container health endpoints."],
            ["Page two explains SQLite write-ahead logging and durable transactions."],
        ],
        output_path=pdf_path,
    )
    chunks = ingest_file(pdf_path, chunk_size=300, chunk_overlap=40)
    assert len(chunks) == 2
    assert chunks[0].source_file == "sample_notes.pdf"
    assert chunks[0].page == 1
    assert "Kubernetes" in chunks[0].text
    assert chunks[1].page == 2
    assert "SQLite" in chunks[1].text
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_ingest_directory_loads_md_txt_and_pdf() -> None:
    chunks = ingest_directory(CORPUS_DIR, chunk_size=480, chunk_overlap=80)
    sources = {c.source_file for c in chunks}
    assert "01_faiss_architecture.md" in sources
    assert "02_bm25_and_rrf.md" in sources
    assert "03_rag_grounding_and_citations.txt" in sources
    assert "04_fastapi_production_ops.md" in sources
    assert "05_python_async_internals.pdf" in sources
    pdf_pages = {c.page for c in chunks if c.source_file == "05_python_async_internals.pdf"}
    assert pdf_pages == {1, 2}


def test_local_embedder_is_deterministic_and_l2_normalized() -> None:
    docs = [
        "FAISS IndexFlatIP computes inner product over unit vectors.",
        "Okapi BM25 scores lexical keyword matches with length normalization.",
        "Python asyncio schedules coroutines on a single-threaded event loop.",
    ]
    e1 = LocalHybridEmbedder(svd_dim=32, hash_dim=64)
    e2 = LocalHybridEmbedder(svd_dim=32, hash_dim=64)
    v1 = e1.fit_transform(docs)
    v2 = e2.fit_transform(docs)

    assert v1.shape == (3, 96)
    assert v1.dtype == np.float32
    np.testing.assert_allclose(v1, v2, atol=1e-6)
    norms = np.linalg.norm(v1, axis=1)
    np.testing.assert_allclose(norms, np.ones(3, dtype=np.float32), atol=1e-5)


def test_openai_compatible_embedder_via_mock_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-ephemeral-key-not-real")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        assert request.headers["Authorization"] == "Bearer test-ephemeral-key-not-real"
        body = json.loads(request.content.decode("utf-8"))
        inputs = body["input"]
        data = [{"index": i, "embedding": [3.0, 4.0, 0.0, float(i)]} for i in range(len(inputs))]
        return httpx.Response(200, json={"data": data})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = OpenAICompatibleEmbedder(http_client=client)
    vecs = embedder.fit_transform(["alpha", "beta"])
    assert vecs.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), [1.0, 1.0], atol=1e-5)

    # Secrets must never be written into serialized embedder state
    state = embedder.to_state()
    assert "api_key" not in state
    assert "test-ephemeral-key-not-real" not in json.dumps(state)


def test_create_embedder_defaults_to_local_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = create_embedder("auto")
    assert isinstance(embedder, LocalHybridEmbedder)


def test_bm25_ranks_exact_lexical_match_first() -> None:
    bm25 = BM25Index().fit(
        [
            "General networking guide for load balancers.",
            "FAISS IndexFlatIP uses faiss.write_index and faiss.read_index for disk persistence.",
            "Database migration strategies for relational schemas.",
        ]
    )
    scores = bm25.score_all("faiss.write_index persistence")
    assert int(np.argmax(scores)) == 1
    assert scores[1] > 0.0
    assert scores[0] == 0.0


def test_hybrid_store_uses_faiss_indexflatip_and_rrf() -> None:
    chunks = ingest_directory(CORPUS_DIR, chunk_size=480, chunk_overlap=80)
    store = HybridVectorStore(embedder=LocalHybridEmbedder())
    count = store.build(chunks)

    assert count == len(chunks)
    assert isinstance(store.index, faiss.IndexFlatIP)
    assert store.index.ntotal == len(chunks)

    hits = store.search("Reciprocal Rank Fusion formula k 60", top_k=3, mode="hybrid")
    assert len(hits) == 3
    assert hits[0].chunk.source_file == "02_bm25_and_rrf.md"
    assert hits[0].rrf_score > 0.0
    assert hits[0].dense_score > 0.0
    assert hits[0].bm25_score > 0.0


def test_persist_and_reload_yields_identical_search_results(tmp_path: Path) -> None:
    chunks = ingest_directory(CORPUS_DIR, chunk_size=480, chunk_overlap=80)
    store = HybridVectorStore(embedder=LocalHybridEmbedder())
    store.build(chunks)

    index_dir = tmp_path / "persisted_index"
    faiss_path, chunks_path = store.save(index_dir)
    assert faiss_path.exists()
    assert chunks_path.exists()

    reloaded = HybridVectorStore.load(index_dir)
    query = "How does FAISS IndexFlatIP compute cosine similarity?"
    before = store.search(query, top_k=3, mode="hybrid")
    after = reloaded.search(query, top_k=3, mode="hybrid")

    assert [r.chunk.chunk_id for r in before] == [r.chunk.chunk_id for r in after]
    for b, a in zip(before, after):
        assert pytest.approx(b.dense_score, abs=1e-4) == a.dense_score
        assert pytest.approx(b.bm25_score, abs=1e-4) == a.bm25_score
        assert pytest.approx(b.rrf_score, abs=1e-6) == a.rrf_score


def test_grounded_qa_includes_valid_inline_citations() -> None:
    chunks = ingest_directory(CORPUS_DIR, chunk_size=480, chunk_overlap=80)
    store = HybridVectorStore(embedder=LocalHybridEmbedder())
    store.build(chunks)
    engine = GroundedQAEngine(store=store, llm_mode="extractive")

    resp = engine.ask("How does Python asyncio handle CancelledError during graceful shutdown?")
    assert resp.refused is False
    assert resp.confidence > 0.3
    assert len(resp.citations) >= 1
    assert any(c.source_file == "05_python_async_internals.pdf" for c in resp.citations)

    inline_tags = re.findall(r"\[[^\[\]]+#[^\[\]]+\]", resp.answer)
    assert len(inline_tags) >= 1
    cited_tags = {c.tag for c in resp.citations}
    assert set(inline_tags).issubset(cited_tags)


def test_grounded_qa_anti_hallucination_refusal() -> None:
    chunks = ingest_directory(CORPUS_DIR, chunk_size=480, chunk_overlap=80)
    store = HybridVectorStore(embedder=LocalHybridEmbedder())
    store.build(chunks)
    engine = GroundedQAEngine(store=store, llm_mode="extractive")

    resp = engine.ask("What temperature bakes sourdough brioche croissants in Paris?")
    assert resp.refused is True
    assert resp.answer == REFUSAL_MESSAGE
    assert resp.citations == []
    assert resp.mode == "refusal"


def test_eval_suite_meets_high_recall_mrr_and_refusal_accuracy(tmp_path: Path) -> None:
    chunks = ingest_directory(CORPUS_DIR, chunk_size=480, chunk_overlap=80)
    store = HybridVectorStore(embedder=LocalHybridEmbedder())
    store.build(chunks)

    out_path = tmp_path / "rag_eval.json"
    report = evaluate_rag(store=store, top_k=3, output_path=out_path)

    assert out_path.exists()
    metrics = report["metrics"]
    assert metrics["recall_at_3"] >= 0.9
    assert metrics["mrr"] >= 0.8
    assert metrics["grounded_citation_accuracy"] >= 0.9
    assert metrics["anti_hallucination_refusal_accuracy"] == 1.0


def test_fastapi_endpoints_health_ingest_and_ask(tmp_path: Path) -> None:
    app = create_app(index_dir=tmp_path / "api_index")
    client = TestClient(app)

    h0 = client.get("/health")
    assert h0.status_code == 200
    assert h0.json()["index_loaded"] is False

    ing = client.post(
        "/ingest",
        json={
            "path": str(CORPUS_DIR),
            "chunk_size": 480,
            "chunk_overlap": 80,
            "backend": "local",
        },
    )
    assert ing.status_code == 200
    assert ing.json()["num_sources"] == 5
    assert ing.json()["num_chunks"] >= 5

    h1 = client.get("/health")
    assert h1.status_code == 200
    assert h1.json()["index_loaded"] is True
    assert h1.json()["num_chunks"] == ing.json()["num_chunks"]

    ask_ok = client.post(
        "/ask",
        json={"query": "What do k1 and b control in Okapi BM25?", "top_k": 3},
    )
    assert ask_ok.status_code == 200
    body_ok = ask_ok.json()
    assert body_ok["refused"] is False
    assert any(c["source_file"] == "02_bm25_and_rrf.md" for c in body_ok["citations"])

    ask_refuse = client.post(
        "/ask",
        json={"query": "Who won the 1994 FIFA World Cup penalty shootout?", "top_k": 3},
    )
    assert ask_refuse.status_code == 200
    assert ask_refuse.json()["refused"] is True


def test_cli_ingest_ask_eval_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_dir = tmp_path / "cli_idx"
    report_path = tmp_path / "cli_eval.json"

    rc_ing = cli_main(
        ["ingest", str(CORPUS_DIR), "--index-dir", str(index_dir), "--backend", "local"]
    )
    assert rc_ing == 0
    ing_out = json.loads(capsys.readouterr().out)
    assert ing_out["status"] == "indexed"
    assert ing_out["num_sources"] == 5

    rc_ask = cli_main(
        [
            "ask",
            "Which FAISS functions persist and reload an index from disk?",
            "--index-dir",
            str(index_dir),
            "--llm-mode",
            "extractive",
            "--json",
        ]
    )
    assert rc_ask == 0
    ask_out = json.loads(capsys.readouterr().out)
    assert ask_out["refused"] is False
    assert ask_out["citations"]

    rc_eval = cli_main(
        [
            "eval",
            "--corpus",
            str(CORPUS_DIR),
            "--index-dir",
            str(index_dir),
            "--output",
            str(report_path),
            "--backend",
            "local",
        ]
    )
    assert rc_eval == 0
    assert report_path.exists()
    capsys.readouterr()


def test_ingest_raw_document_unsupported_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "binary.exe"
    bad_file.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        ingest_file(bad_file)

    raw_chunks = ingest_raw_document("Short note about FAISS.", source_file="note.txt")
    assert len(raw_chunks) == 1
    assert raw_chunks[0].citation_tag == "[note.txt#note_p1_c000]"
