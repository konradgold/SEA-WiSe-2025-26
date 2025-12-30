from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sea.index.tokenization import TokenizerAbstract, get_tokenizer
from sea.ranking.io_wrapper import bm25 as build_bm25_ranker
from sea.ranking.utils import Document
from sea.utils.config import Config


@dataclass
class BM25Retriever:
    """
    Thin wrapper around the BM25 implementation for use in LTR pipelines for LTR candidate generation.
    """

    cfg: Config
    tokenizer: TokenizerAbstract
    ranker: object

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "BM25Retriever":
        cfg = cfg or Config(load=True)
        tokenizer = get_tokenizer(cfg)
        ranker = build_bm25_ranker()
        return cls(cfg=cfg, tokenizer=tokenizer, ranker=ranker)

    def retrieve(self, query: str, *, topn: int) -> list[Document]:
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        # Override limit of top-n candidates since the other BM25 ranking truncates to `cfg.SEARCH.MAX_RESULTS`
        inner_ranker = self.ranker.ranker
        old_max = inner_ranker.max_results
        inner_ranker.max_results = topn
        try:
            docs: list[Document] = self.ranker(tokens)
        finally:
            inner_ranker.max_results = old_max

        return docs[:topn] if topn > 0 else []

    def retrieve_ids(self, query: str, *, topn: int) -> list[tuple[int, float]]:
        """Returns only (doc_id, score) tuples without reading document content from disk."""
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        token_list = self.ranker._prepare_tokens(tokens)
        if not token_list:
            return []

        inner_ranker = self.ranker.ranker
        old_max = inner_ranker.max_results
        inner_ranker.max_results = topn
        try:
            results = inner_ranker.rank(token_list)
        finally:
            inner_ranker.max_results = old_max

        return results

    def hydrate_docs(self, id_score_pairs: list[tuple[int, float]]) -> list[Document]:
        """Reads document content from disk for a specific set of IDs."""
        return self.ranker._read_documents(id_score_pairs)

    def retrieve_many(self, queries: Iterable[tuple[int, str]], *, topn: int):
        for qid, qtext in queries:
            yield qid, self.retrieve(qtext, topn=topn)
