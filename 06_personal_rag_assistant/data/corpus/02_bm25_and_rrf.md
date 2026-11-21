# Hybrid Retrieval: Okapi BM25 and Reciprocal Rank Fusion (RRF)

Dense embeddings capture topical similarity and paraphrasing, while lexical sparse retrieval excels on exact identifiers, error codes, acronyms, and rare technical terms. Hybrid search combines both signals to maximize Recall@K and Mean Reciprocal Rank (MRR).

## Okapi BM25 Hyperparameters

Okapi BM25 scores documents by combining inverse document frequency (IDF) with saturated term frequency and document length normalization. The hyperparameter `k1` (typically `1.2` to `2.0`, default `1.5`) controls term-frequency saturation so repeated keywords yield diminishing returns. The hyperparameter `b` (typically `0.75`) controls document length normalization, penalizing overly long passages relative to the average chunk length.

## Reciprocal Rank Fusion (RRF)

Because raw FAISS cosine scores and unbounded BM25 scores live on incompatible numerical scales, Reciprocal Rank Fusion (RRF) merges ranked lists purely by rank position. The RRF formula is `score(d) = sum_i 1 / (k + rank_i(d))` with default smoothing constant `k = 60`. Documents ranked near the top in both dense FAISS retrieval and sparse BM25 retrieval rise cleanly to the top of the fused ranking.
