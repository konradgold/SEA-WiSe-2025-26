from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sea.index.tokenization import TokenizerAbstract, get_tokenizer
from sea.ranking.io_wrapper import RankerAdapter, bm25 as build_bm25_ranker
from sea.ranking.utils import Document
from sea.utils.config_wrapper import Config
from omegaconf import DictConfig


@dataclass
class BM25Retriever:
    """Wrapper around BM25 for LTR candidate generation.

    This allows retrieving more candidates for reranking.
    """

    config: DictConfig
    tokenizer: TokenizerAbstract
    ranker: RankerAdapter

    @classmethod
    def from_config(cls, config: DictConfig | None = None, num_threads: int | None = None) -> "BM25Retriever":
        config = config or Config(load=True)
        tokenizer = get_tokenizer(config)
        ranker = build_bm25_ranker(config, num_threads=num_threads)
        return cls(config=config, tokenizer=tokenizer, ranker=ranker)

    def retrieve(self, query: str, *, topn: int) -> list[Document]:
        """Retrieve top-N documents with full content loaded from disk."""
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        # Temporarily override the result limit to get more candidates
        original_max_results = self.ranker.max_results
        self.ranker.max_results = topn
        try:
            documents: list[Document] = self.ranker(tokens)
        finally:
            self.ranker.max_results = original_max_results

        return documents[:topn] if topn > 0 else []

    def retrieve_ids(self, query: str, *, topn: int) -> list[tuple[int, float]]:
        """Retrieve only (document_id, score) tuples without reading content from disk."""
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        accumulated_scores: dict[int, float] = dict()
        for field in self.ranker.fields:
            prepared_tokens = self.ranker._prepare_tokens(tokens, field=field)
            if not prepared_tokens:
                continue
            for document_id, score in self.ranker.ranker(prepared_tokens, field=field).items():
                accumulated_scores[document_id] = accumulated_scores.get(document_id, 0.0) + score

        if not accumulated_scores:
            return []

        sorted_results = sorted(
            accumulated_scores.items(), key=lambda item: item[1], reverse=True
        )[:topn]
        return sorted_results

    def hydrate_docs(self, id_score_pairs: list[tuple[int, float]]) -> list[Document]:
        """Load document content from disk for a specific set of document IDs
        """
        return self.ranker._read_documents(id_score_pairs)

    def retrieve_many(self, queries: Iterable[tuple[int, str]], *, topn: int):
        """Retrieve documents for multiple queries"""
        for query_id, query_text in queries:
            yield query_id, self.retrieve(query_text, topn=topn)
