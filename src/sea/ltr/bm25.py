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
    """
    Thin wrapper around the BM25 implementation for use in LTR pipelines for LTR candidate generation.
    """

    cfg: DictConfig
    tokenizer: TokenizerAbstract
    ranker: RankerAdapter

    @classmethod
    def from_config(cls, cfg: DictConfig | None = None, num_threads: int | None = None) -> "BM25Retriever":
        cfg = cfg or Config(load=True)
        tokenizer = get_tokenizer(cfg)
        ranker = build_bm25_ranker(cfg, num_threads=num_threads)
        return cls(cfg=cfg, tokenizer=tokenizer, ranker=ranker)

    def retrieve(self, query: str, *, topn: int) -> list[Document]:
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        # Override limit of top-n candidates since the other BM25 ranking truncates to `cfg.SEARCH.MAX_RESULTS`
        old_max = self.ranker.max_results
        self.ranker.max_results = topn
        try:
            docs: list[Document] = self.ranker(tokens)
        finally:
            self.ranker.max_results = old_max

        return docs[:topn] if topn > 0 else []

    def retrieve_ids(self, query: str, *, topn: int) -> list[tuple[int, float]]:
        """Returns only (doc_id, score) tuples without reading document content from disk."""
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        ranked_results: dict[int, float] = dict()
        for field in self.ranker.fields:
            token_list = self.ranker._prepare_tokens(tokens, field=field)
            if not token_list:
                continue
            for doc_id, score in self.ranker.ranker(token_list, field=field).items():
                ranked_results[doc_id] = ranked_results.get(doc_id, 0.0) + score

        if not ranked_results:
            return []

        results = sorted(
            ranked_results.items(), key=lambda item: item[1], reverse=True
        )[:topn]
        return results

    def hydrate_docs(self, id_score_pairs: list[tuple[int, float]]) -> list[Document]:
        """Reads document content from disk for a specific set of IDs."""
        return self.ranker._read_documents(id_score_pairs)

    def retrieve_many(self, queries: Iterable[tuple[int, str]], *, topn: int):
        for qid, qtext in queries:
            yield qid, self.retrieve(qtext, topn=topn)
