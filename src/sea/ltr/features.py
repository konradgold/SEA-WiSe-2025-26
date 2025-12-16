from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from sea.index.tokenization import TokenizerAbstract, get_tokenizer
from sea.ranking.utils import Document
from sea.storage.manager import StorageManager
from sea.utils.config import Config


def _num_docs(cfg: Config) -> int:
    # Mirror `sea.ranking.ranking.NUM_DOCS` fallback.
    return int(cfg.SEARCH.NUM_DOCS) if cfg.SEARCH.NUM_DOCS is not None else 3_300_000


@dataclass(frozen=True)
class FeatureSpec:
    """
    Defines the numeric feature vector order.

    Keep this stable across training and serving.
    """

    names: list[str]


DEFAULT_FEATURES = FeatureSpec(
    names=[
        "bm25_score",
        "query_len",
        "query_uniq_len",
        "title_len",
        "body_len",
        "title_overlap_cnt",
        "body_overlap_cnt",
        "body_overlap_ratio",
        "idf_body_overlap_sum",
        "idf_title_overlap_sum",
    ]
)


class _LRUCache:
    """Tiny LRU for tokenized docs to avoid re-tokenizing during reranking."""

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._data: OrderedDict[str, tuple[set[str], int, set[str], int]] = OrderedDict()

    def get(self, key: str):
        if key not in self._data:
            return None
        v = self._data.pop(key)
        self._data[key] = v
        return v

    def put(self, key: str, value):
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)


@dataclass
class FeatureExtractor:
    cfg: Config
    tokenizer: TokenizerAbstract
    storage: StorageManager
    posting_cut: int
    features: FeatureSpec = DEFAULT_FEATURES
    cache_max_docs: int = 10_000

    def __post_init__(self) -> None:
        self._N = float(_num_docs(self.cfg))
        self._cache = _LRUCache(max_size=self.cache_max_docs)

    @classmethod
    def from_config(cls, cfg: Optional[Config] = None, *, cache_max_docs: int = 10_000) -> "FeatureExtractor":
        cfg = cfg or Config(load=True)
        tokenizer = get_tokenizer(cfg)
        storage = StorageManager(rewrite=False, cfg=cfg)
        storage.init_all()
        posting_cut = int(cfg.SEARCH.POSTINGS_CUT) if cfg.SEARCH.POSTINGS_CUT is not None else 100
        return cls(
            cfg=cfg,
            tokenizer=tokenizer,
            storage=storage,
            posting_cut=posting_cut,
            cache_max_docs=cache_max_docs,
        )

    @staticmethod
    def _safe_text(s: Optional[str]) -> str:
        return s if isinstance(s, str) else ""

    def _doc_tokens(self, doc: Document) -> tuple[set[str], int, set[str], int]:
        """
        Returns:
          - title token set, title length
          - body token set, body length
        """
        key = doc.doc_id
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        title = self._safe_text(doc.title)
        body = self._safe_text(doc.content)

        title_tokens = self.tokenizer.tokenize(title)
        body_tokens = self.tokenizer.tokenize(body)

        title_set = set(title_tokens)
        body_set = set(body_tokens)
        value = (title_set, len(title_tokens), body_set, len(body_tokens))
        self._cache.put(key, value)
        return value

    def _idf(self, token: str) -> float:
        """
        BM25-style IDF using index df from the posting list.

        We mimic the repo's BM25 posting cut behavior: if df exceeds cut, treat the term
        as non-contributing (idf=0) to keep feature behavior aligned with retrieval.
        """
        pl = self.storage.getPostingList(token)
        if not pl:
            return 0.0
        # Posting list is [len][docid][tf]...
        if len(pl) < 1:
            return 0.0
        len_pl = int(pl[0])
        if len_pl <= 0:
            return 0.0
        # Without positions: remaining values are (docid, tf) pairs.
        df = max(0, len_pl // 2)
        if df <= 0:
            return 0.0
        if df > self.posting_cut:
            return 0.0
        # Same formula as `sea.ranking.ranking.BM25Ranking`:
        return float(math.log((self._N - df + 0.5) / (df + 0.5) + 1.0))

    def extract(self, query: str, doc: Document) -> np.ndarray:
        q_tokens = self.tokenizer.tokenize(query)
        q_len = len(q_tokens)
        q_set = set(q_tokens)
        q_uniq = len(q_set)

        title_set, title_len, body_set, body_len = self._doc_tokens(doc)

        title_overlap = len(q_set & title_set)
        body_overlap = len(q_set & body_set)
        body_overlap_ratio = body_overlap / max(1, q_uniq)

        idf_body = 0.0
        idf_title = 0.0
        for t in q_set:
            idf = self._idf(t)
            if idf <= 0.0:
                continue
            if t in body_set:
                idf_body += idf
            if t in title_set:
                idf_title += idf

        values = {
            "bm25_score": float(doc.score),
            "query_len": float(q_len),
            "query_uniq_len": float(q_uniq),
            "title_len": float(title_len),
            "body_len": float(body_len),
            "title_overlap_cnt": float(title_overlap),
            "body_overlap_cnt": float(body_overlap),
            "body_overlap_ratio": float(body_overlap_ratio),
            "idf_body_overlap_sum": float(idf_body),
            "idf_title_overlap_sum": float(idf_title),
        }

        x = np.array([values[name] for name in self.features.names], dtype=np.float32)
        return x

    def extract_many(self, query: str, docs: Iterable[Document]) -> np.ndarray:
        return np.stack([self.extract(query, d) for d in docs], axis=0)




