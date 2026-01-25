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
    """Which features to extract for LTR"""
    names: list[str]


def get_default_features(config: Optional[DictConfig] = None) -> FeatureSpec:
    """Get list of features to extract, either from config or defaults"""
    if config and hasattr(config, "LTR") and hasattr(config.LTR, "FEATURES"):
        return FeatureSpec(names=list(config.LTR.FEATURES))
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


class DocumentTokenCache:
    """Cache for tokenized document content. Same document appears in multiple training samples (retrieved for
    different queries) so we cache it
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._data: OrderedDict[str, tuple[set[str], int, set[str], int]] = OrderedDict()

    def get(self, key: str):
        if key not in self._data:
            return None
        # Move to end 
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key: str, value):
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        # Throw out oldest entry if over capacity
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)


@dataclass
class FeatureExtractor:
    """Extracts ranking features from query-document pairs for LTR training and inference"""

    config: DictConfig
    tokenizer: TokenizerAbstract
    storage: StorageManager
    posting_cut: int
    features: FeatureSpec
    cache_max_docs: int = 10_000

    def __post_init__(self) -> None:
        self._total_documents = (
            float(self.config.SEARCH.NUM_DOCS)
            if self.config.SEARCH.NUM_DOCS is not None
            else 3_300_000.0
        )
        self._token_cache = DocumentTokenCache(max_size=self.cache_max_docs)

    @classmethod
    def from_config(cls, config: Optional[DictConfig] = None, *, cache_max_docs: int = 10_000) -> "FeatureExtractor":
        config = config or Config(load=True)
        tokenizer = get_tokenizer(config)

        field = None
        if config.SEARCH.FIELDED.ACTIVE:
            fields = config.SEARCH.FIELDED.FIELDS
            field = "body" if "body" in fields else (fields[0] if fields else None)

        storage = StorageManager(rewrite=False, cfg=config, field=field)
        storage.init_all()
        posting_cut = int(config.SEARCH.POSTINGS_CUT) if config.SEARCH.POSTINGS_CUT is not None else 100
        features = get_default_features(config)
        return cls(
            config=config,
            tokenizer=tokenizer,
            storage=storage,
            posting_cut=posting_cut,
            features=features,
            cache_max_docs=cache_max_docs,
        )

    def _get_document_tokens(self, document: Document) -> tuple[set[str], int, set[str], int]:
        """Get tokenized title and body for a document, using cache if available

        Returns: (title_token_set, title_length, body_token_set, body_length)
        """
        cache_key = document.doc_id
        cached_value = self._token_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        title_text = document.title if isinstance(document.title, str) else ""
        body_text = document.content if isinstance(document.content, str) else ""

        title_tokens = self.tokenizer.tokenize(title_text)
        body_tokens = self.tokenizer.tokenize(body_text)

        result = (
            set(title_tokens),
            len(title_tokens),
            set(body_tokens),
            len(body_tokens),
        )
        self._token_cache.put(cache_key, result)
        return result

    def _compute_idf(self, token: str) -> float:
        """Compute BM25-style IDF for a token using document frequency from the index.

        IDF = log((N - df + 0.5) / (df + 0.5) + 1)

        Returns 0 if token is too common (df > posting_cut) or not in index.
        """
        posting_list = self.storage.getPostingList(token)
        if not posting_list or len(posting_list) < 1:
            return 0.0

        # First element is the count of values in the posting list
        posting_list_length = int(posting_list[0])
        # Each doc has 2 entries (doc_id, term_freq), so df = length / 2
        document_frequency = max(0, posting_list_length // 2)

        # Skip terms that are too common or not found
        if document_frequency <= 0 or document_frequency > self.posting_cut:
            return 0.0

        return float(math.log((self._total_documents - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0))

    def extract(self, query: str, document: Document) -> np.ndarray:
        """Extract features for a single query-document pair"""
        query_tokens = self.tokenizer.tokenize(query)
        return self._extract_with_tokens(query_tokens, document)

    def _extract_with_tokens(
        self,
        query_tokens: list[str],
        document: Document,
        query_idf_cache: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """Extract features for a single query-document pair with pre-computed query IDFs if provided
        """
        query_length = len(query_tokens)
        query_token_set = set(query_tokens)
        query_unique_count = len(query_token_set)

        title_token_set, title_length, body_token_set, body_length = self._get_document_tokens(document)

        title_overlap_count = len(query_token_set & title_token_set)
        body_overlap_count = len(query_token_set & body_token_set)
        body_overlap_ratio = body_overlap_count / max(1, query_unique_count)

        # Sum IDF weights for query terms that appear in title/body
        idf_weighted_body_overlap = 0.0
        idf_weighted_title_overlap = 0.0
        for token in query_token_set:
            token_idf = query_idf_cache[token] if query_idf_cache is not None else self._compute_idf(token)
            if token_idf <= 0.0:
                continue
            if token in body_token_set:
                idf_weighted_body_overlap += token_idf
            if token in title_token_set:
                idf_weighted_title_overlap += token_idf

        feature_values = {
            "bm25_score": float(document.score),
            "query_len": float(query_length),
            "query_uniq_len": float(query_unique_count),
            "title_len": float(title_length),
            "body_len": float(body_length),
            "title_overlap_cnt": float(title_overlap_count),
            "body_overlap_cnt": float(body_overlap_count),
            "body_overlap_ratio": float(body_overlap_ratio),
            "idf_body_overlap_sum": float(idf_weighted_body_overlap),
            "idf_title_overlap_sum": float(idf_weighted_title_overlap),
        }

        return np.array(
            [feature_values[name] for name in self.features.names], dtype=np.float32
        )

    def extract_many(self, query: str, documents: Iterable[Document]) -> np.ndarray:
        """Extract features for multiple documents with the same query
        """
        query_tokens = self.tokenizer.tokenize(query)
        query_idf_cache = {token: self._compute_idf(token) for token in set(query_tokens)}
        return np.stack(
            [self._extract_with_tokens(query_tokens, document, query_idf_cache) for document in documents], axis=0
        )
