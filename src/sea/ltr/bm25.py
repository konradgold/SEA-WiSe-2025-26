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
    Thin wrapper around the repo's BM25 implementation for use in LTR pipelines.

    Important: We do NOT use `sea.query.search.search_documents()` because it:
    - goes through query operators (AND/OR) with unsorted dict merges, and
    - sorts ascending in one place.

    For LTR candidate generation we want a stable, deterministic top-N based on BM25 score.
    """

    cfg: Config
    tokenizer: TokenizerAbstract
    ranker: object  # `sea.ranking.io_wrapper.BM25` adapter

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
        # Critical: the underlying BM25 ranking truncates to `cfg.SEARCH.MAX_RESULTS` at construction time.
        # For LTR we need top-N candidates (often N >> 10), so we must override that limit here.
        try:
            # `self.ranker` is the RankerAdapter; `self.ranker.ranker` is the BM25Ranking instance.
            if hasattr(self.ranker, "ranker") and hasattr(self.ranker.ranker, "max_results"):
                self.ranker.ranker.max_results = max(int(topn), int(getattr(self.ranker.ranker, "max_results", topn)))
        except Exception:
            # If for some reason we can't override, we still return what we got (better than crashing).
            pass
        # The adapter already returns documents sorted by BM25 score descending.
        docs: list[Document] = self.ranker(tokens)
        if topn <= 0:
            return []
        if len(docs) <= topn:
            return docs
        return docs[:topn]

    def retrieve_ids(self, query: str, *, topn: int) -> list[tuple[int, float]]:
        """Returns only (doc_id, score) tuples without reading document content from disk."""
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        # Reach into the RankerAdapter to prepare tokens and call rank() directly
        token_list = self.ranker._prepare_tokens(tokens)
        if not token_list:
            return []

        # Use the underlying BM25Ranking.rank method which returns (id, score) pairs
        # We need to temporarily set max_results on the inner ranker
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
