"""Brute-force semantic search over pre-computed embeddings."""

from time import perf_counter
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from sea.semantic.client import EmbeddingClient
from sea.storage.embeddings import EmbeddingIO
from sea.utils.config_wrapper import Config


class SemanticSearcher:
    """Similarity search using document embeddings"""

    def __init__(self, cfg: Optional[DictConfig] = None, verbose: bool = False):
        config = cfg if cfg is not None else Config(load=True)
        self.config = config
        self.verbose = verbose

        # Load all document embeddings into memory
        self.embedding_io = EmbeddingIO(config)
        self.corpus = self.embedding_io.load_all()

        # Client for computing query embeddings via service
        self.client = EmbeddingClient(base_url=config.SEMANTIC.SERVICE_URL)

        if self.verbose:
            size_megabytes = self.corpus.nbytes / (1024 * 1024)
            print(f"Loaded {self.corpus.shape[0]:,} embeddings ({self.corpus.shape[1]} dims, {size_megabytes:.1f} MB)")

    def search(self, query: str, topn: int) -> list[tuple[int, float]]:
        """Search for similar documents using cosine similarity.

        Returns list of (document_id, similarity_score) tuples sorted by score descending.
        document_id is the row index (0-based) matching the corpus embedding index.
        """
        time_start = perf_counter()

        query_embedding = self.client.embed_query(query)

        time_after_embedding = perf_counter()

        # Query and corpus embeddings are normalized so
        # dot product equals cosine similarity
        similarity_scores = self.corpus @ query_embedding

        # Top-k selection
        if topn >= len(similarity_scores):
            top_indices = np.argsort(-similarity_scores)
        else:
            top_indices = np.argpartition(-similarity_scores, topn)[:topn]
            # Sort only the top-k for final ordering
            top_indices = top_indices[np.argsort(-similarity_scores[top_indices])]

        time_after_search = perf_counter()

        if self.verbose:
            embedding_time_ms = (time_after_embedding - time_start) * 1000
            search_time_ms = (time_after_search - time_after_embedding) * 1000
            print(f"Semantic: embed={embedding_time_ms:.1f}ms, search={search_time_ms:.2f}ms")

        return [(int(index), float(similarity_scores[index])) for index in top_indices]
