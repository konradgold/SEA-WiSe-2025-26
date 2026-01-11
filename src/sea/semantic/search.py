"""Brute-force semantic search over pre-computed embeddings."""

from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from sea.semantic.client import EmbeddingClient
from sea.storage.embeddings import EmbeddingIO
from sea.utils.config_wrapper import Config


class SemanticSearcher:
    def __init__(self, cfg: Optional[DictConfig] = None, verbose: bool = False):
        if cfg is None:
            cfg = Config(load=True)
        self.cfg = cfg
        self.verbose = verbose

        # Load corpus embeddings
        self.embedding_io = EmbeddingIO(cfg)
        self.corpus = self.embedding_io.load_all()

        # Client for query embeddings
        self.client = EmbeddingClient(base_url=cfg.SEMANTIC.SERVICE_URL)

        if self.verbose:
            mb = self.corpus.nbytes / (1024 * 1024)
            print(f"Loaded {self.corpus.shape[0]:,} embeddings ({self.corpus.shape[1]} dims, {mb:.1f} MB)")

    def search(self, query: str, topn: int) -> list[tuple[int, float]]:
        """
        Search for similar documents using cosine similarity.

        Returns list of (doc_id, score) tuples sorted by score descending.
        doc_id is the row index (0-based) matching the corpus embedding index.
        """
        t0 = perf_counter()

        # Get query embedding from service
        q_emb = self.client.embed_query(query)

        t1 = perf_counter()

        # Brute-force: matrix-vector product (both normalized, so this is cosine sim)
        scores = self.corpus @ q_emb

        # Get top-n indices
        if topn >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, topn)[:topn]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        t2 = perf_counter()

        if self.verbose:
            embed_ms = (t1 - t0) * 1000
            search_ms = (t2 - t1) * 1000
            print(f"Semantic: embed={embed_ms:.1f}ms, search={search_ms:.2f}ms")

        return [(int(i), float(scores[i])) for i in top_idx]
