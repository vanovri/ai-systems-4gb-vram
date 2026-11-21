# Project #6 - Personal RAG Assistant (`06_personal_rag_assistant`)

Production-grade local-first Retrieval-Augmented Generation (RAG) assistant supporting `.pdf` (`pypdf`), `.md`, and `.txt` ingestion, recursive character/sentence chunking, **FAISS `IndexFlatIP`** dense retrieval + **Okapi BM25** lexical search fused via **Reciprocal Rank Fusion (RRF)**, disk persistence (`index.faiss` + `chunks.json`), strict inline `[source#chunk_id]` citation enforcement, anti-hallucination refusal, CLI, and FastAPI service.

## Architecture

1. **Document Ingestion (`rag/ingest.py`)**:
   - Extracts text from `.txt`, `.md`, and multi-page `.pdf` (`pypdf.PdfReader`).
   - Recursive character/sentence chunking with configurable `chunk_size` and `chunk_overlap` preserving full provenance metadata (`source_file`, `page`, `chunk_id`, `start_char`, `end_char`).
2. **Dual-Mode Dense Embeddings (`rag/embeddings.py`)**:
   - **Offline Default (`LocalHybridEmbedder`)**: Deterministic TF-IDF + Truncated SVD combined with signed blake2b character 3–5 gram & word unigram/bigram hashing, L2-normalized to unit length in `float32` so FAISS `IndexFlatIP` computes exact cosine similarity with **zero API key and zero GPU**.
   - **Live OpenAI-Compatible Mode (`OpenAICompatibleEmbedder`)**: Automatically activates when `OPENAI_API_KEY` is provided via environment variables (zero hardcoded secrets; keys are never serialized to disk).
3. **Hybrid Retrieval Engine (`rag/store.py`)**:
   - Real `faiss.IndexFlatIP` binary vector index (`faiss.write_index` / `faiss.read_index`).
   - Pure Okapi BM25 lexical scorer (`k1=1.5`, `b=0.75`).
   - **Reciprocal Rank Fusion (RRF)**: `score(d) = 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)`.
   - Persists & reloads cleanly from `index.faiss` + `chunks.json`.
4. **Grounded QA & Anti-Hallucination Refusal (`rag/qa.py`)**:
   - Enforces inline `[source_file#chunk_id]` citations mapped directly to retrieved context chunks.
   - Calibrated grounding check emits an explicit anti-hallucination refusal when queries are unsupported by indexed documents.
5. **CLI & FastAPI (`rag/cli.py`, `rag/api.py`)**:
   - CLI subcommands: `ingest`, `ask`, `eval`, `serve`.
   - REST endpoints: `GET /health`, `POST /ingest`, `POST /ask`.

## Verified Evaluation Metrics (`reports/rag_eval.json`)

Evaluated across the 5-document sample corpus (`01_faiss_architecture.md`, `02_bm25_and_rrf.md`, `03_rag_grounding_and_citations.txt`, `04_fastapi_production_ops.md`, and 2-page `05_python_async_internals.pdf` - 16 indexed chunks, 256-dim `faiss.IndexFlatIP`):

| Metric | Value | Notes |
| :--- | ---: | :--- |
| Answerable Queries | 10 | Covers all 5 MD/TXT/PDF files |
| Out-of-Scope Refusal Queries | 2 | Unsupported factual queries |
| **Recall@1** | **1.0000** | 10/10 exact top-1 document hit |
| **Recall@3** | **1.0000** | 10/10 top-3 hit |
| **MRR (Mean Reciprocal Rank)** | **1.0000** | Hybrid RRF (`dense + BM25`) |
| **Grounded Citation Accuracy** | **1.0000** | 10/10 valid inline `[source#chunk_id]` citations |
| **Anti-Hallucination Refusal Accuracy** | **1.0000** | 2/2 clean abstentions, 0 fabricated citations |

## Project Structure

```text
06_personal_rag_assistant/
├── rag/
│   ├── __init__.py
│   ├── models.py          # DocumentChunk, SearchResult, Citation, QAResponse
│   ├── ingest.py          # .txt/.md/.pdf ingestion + recursive chunker
│   ├── embeddings.py      # LocalHybridEmbedder (TF-IDF/SVD+n-gram) & OpenAICompatibleEmbedder
│   ├── store.py           # FAISS IndexFlatIP + BM25 + RRF + save/load
│   ├── qa.py              # Grounded QA + inline citations + anti-hallucination refusal
│   ├── eval.py            # Recall@3, MRR, citation & refusal evaluator
│   ├── api.py             # FastAPI app (/health, /ingest, /ask)
│   └── cli.py             # CLI entrypoint (ingest, ask, eval, serve)
├── data/
│   ├── corpus/            # 5 technical docs (.md, .txt, multi-page .pdf)
│   └── index/             # Persisted index.faiss + chunks.json
├── reports/
│   └── rag_eval.json      # Measured retrieval & grounding report
├── scripts/
│   └── generate_sample_pdf.py
├── tests/
│   └── test_rag.py        # 15 pytest tests (ingest, PDF, FAISS, RRF, persist, QA, FastAPI, CLI)
├── pytest.ini
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Ingest sample corpus (.md, .txt, .pdf) into persistent FAISS index
python -m rag.cli ingest data/corpus --index-dir data/index --backend local

# 2. Ask grounded questions with inline citations (or test refusal on out-of-scope queries)
python -m rag.cli ask "How does FAISS IndexFlatIP compute cosine similarity?" --index-dir data/index
python -m rag.cli ask "What temperature bakes sourdough brioche croissants in Paris?" --index-dir data/index

# 3. Run retrieval & grounding benchmark -> reports/rag_eval.json
python -m rag.cli eval --corpus data/corpus --index-dir data/index --output reports/rag_eval.json --rebuild

# 4. Serve FastAPI REST API (/health, /ingest, /ask)
python -m rag.cli serve --index-dir data/index --host 127.0.0.1 --port 8000

# 5. Run full test suite (15 tests)
pytest -v
```

### Live OpenAI Mode (Optional)

Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_CHAT_MODEL`) in your environment to switch `--backend auto` to OpenAI-compatible embeddings and chat synthesis with strict citation validation.
