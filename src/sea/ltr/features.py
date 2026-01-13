from __future__ import annotations
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from omegaconf import DictConfig

from sea.index.tokenization import TokenizerAbstract, get_tokenizer
from sea.ranking.utils import Document
from sea.storage.manager import StorageManager
from sea.utils.config_wrapper import Config


@dataclass(frozen=True)
class FeatureSpec:
    names: list[str]


def get_default_features(cfg: Optional[DictConfig] = None) -> FeatureSpec:
    if cfg and hasattr(cfg, "LTR") and hasattr(cfg.LTR, "FEATURES"):
        return FeatureSpec(names=list(cfg.LTR.FEATURES))
    return FeatureSpec(
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
    cfg: DictConfig
    tokenizer: TokenizerAbstract
    storage_dict: dict[str, StorageManager]
    posting_cut: int
    features: FeatureSpec
    cache_max_docs: int = 10_000

    def __post_init__(self) -> None:
        self._N = (
            float(self.cfg.SEARCH.NUM_DOCS)
            if self.cfg.SEARCH.NUM_DOCS is not None
            else 3_300_000.0
        )
        self._cache = _LRUCache(max_size=self.cache_max_docs)

    @classmethod
    def from_config(cls, cfg: Optional[DictConfig] = None, *, cache_max_docs: int = 10_000) -> "FeatureExtractor":
        cfg = cfg or Config(load=True)
        tokenizer = get_tokenizer(cfg)
        fields: list[str] = ["all"] if not cfg.SEARCH.FIELDED.ACTIVE else cfg.SEARCH.FIELDED.FIELDS
        storage_dict: dict[str, StorageManager] = {}
        for field in fields:
            storage = StorageManager(rewrite=False, cfg=cfg,
                                     field=field)
            storage.init_all()
            storage_dict[field] = storage
        posting_cut = int(cfg.SEARCH.POSTINGS_CUT) if cfg.SEARCH.POSTINGS_CUT is not None else 100
        features = get_default_features(cfg)
        return cls(
            cfg=cfg,
            tokenizer=tokenizer,
            storage_dict=storage_dict,
            posting_cut=posting_cut,
            features=features,
            cache_max_docs=cache_max_docs,
        )

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

        title = doc.title if isinstance(doc.title, str) else ""
        body = doc.content if isinstance(doc.content, str) else ""

        title_tokens = self.tokenizer.tokenize(title)
        body_tokens = self.tokenizer.tokenize(body)

        value = (
            set(title_tokens),
            len(title_tokens),
            set(body_tokens),
            len(body_tokens),
        )
        self._cache.put(key, value)
        return value

    def _idf(self, token: str) -> float:
        """
        BM25-style IDF using index df from the posting list.
        """
        len_pl = 0
        any_pl = False
        for field, storage in self.storage_dict.items():
            pl = storage.getPostingList(token)
            if pl and len(pl) >= 1:
                any_pl = True
            else:
                continue
        
            len_pl += int(pl[0])
        if not any_pl:
            return 0.0
        df = max(0, len_pl // 2)
        if df <= 0 or df > self.posting_cut:
            return 0.0

        return float(math.log((self._N - df + 0.5) / (df + 0.5) + 1.0))

    def extract(self, query: str, doc: Document) -> np.ndarray:
        q_tokens = self.tokenizer.tokenize(query)
        return self._extract_with_tokens(q_tokens, doc)

    def _extract_with_tokens(
        self,
        q_tokens: list[str],
        doc: Document,
        q_idfs: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
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
            idf = q_idfs[t] if q_idfs is not None else self._idf(t)
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

        return np.array(
            [values[name] for name in self.features.names], dtype=np.float32
        )

    def extract_many(self, query: str, docs: Iterable[Document]) -> np.ndarray:
        q_tokens = self.tokenizer.tokenize(query)
        q_idfs = {t: self._idf(t) for t in set(q_tokens)}
        return np.stack(
            [self._extract_with_tokens(q_tokens, d, q_idfs) for d in docs], axis=0
        )
