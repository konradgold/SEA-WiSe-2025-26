"""
Learning-to-rank (LTR) reranking components.

This package is intentionally separated from the lexical retrieval code:
- Retrieval (BM25) lives under `sea.ranking` / `sea.query`.
- LTR builds training data from BM25 candidates, computes features, trains a model,
  and provides a serving-time reranker.
"""




