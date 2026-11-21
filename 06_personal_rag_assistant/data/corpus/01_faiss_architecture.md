# FAISS Vector Store Architecture

FAISS (Facebook AI Similarity Search) provides fast dense vector retrieval on CPU and GPU. For personal knowledge bases containing thousands to tens of thousands of chunks, exact search with `IndexFlatIP` offers zero approximation error and sub-millisecond query latency.

## Cosine Similarity with IndexFlatIP

`faiss.IndexFlatIP` computes the exact inner product between a query vector and every indexed document vector. When both document vectors and query vectors are L2-normalized to unit length (`||v||_2 = 1.0`) in `float32`, the inner product equals the exact cosine similarity in `[-1.0, 1.0]`.

## Persistence and Reload

A production RAG store must persist both the dense binary index and the chunk metadata catalog. FAISS exports indexes using `faiss.write_index(index, "index.faiss")` and reloads them deterministically via `faiss.read_index("index.faiss")`. Paired with a `chunks.json` manifest storing chunk text, source filenames, page numbers, and embedder state, reloading restores identical vector similarity rankings across restarts.
