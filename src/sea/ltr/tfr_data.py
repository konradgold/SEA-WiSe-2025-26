from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.candidates import CandidateDoc
from sea.ltr.features import FeatureExtractor
from sea.ranking.utils import Document


@dataclass(frozen=True)
class ListwiseSample:
    """
    One query produces a fixed-size list of documents:
      - exactly one positive (if available in candidates)
      - `num_neg` negatives sampled from remaining candidates
    """

    qid: int
    features: np.ndarray  # [list_size, num_features] float32
    labels: np.ndarray  # [list_size] float32


def _sample_list_for_query(
    *,
    qid: int,
    query: str,
    docs: list[Document],
    positives: set[str],
    fe: FeatureExtractor,
    list_size: int,
    seed: int,
    hard_pool_topk: int = 50,
) -> Optional[ListwiseSample]:
    if list_size < 2:
        raise ValueError("list_size must be >= 2")

    # Identify candidates and positives within retrieved set
    pos_docs = [d for d in docs if d.doc_id in positives]
    if not pos_docs:
        return None

    # Pick one positive (deterministic-ish but seedable)
    rng = random.Random((seed * 1_000_003) ^ qid)
    pos = rng.choice(pos_docs)

    neg_docs = [d for d in docs if d.doc_id not in positives]
    if not neg_docs:
        return None

    pool = neg_docs[: max(1, min(hard_pool_topk, len(neg_docs)))]
    num_neg = list_size - 1
    negs = [rng.choice(pool) for _ in range(num_neg)]

    # Build list: (pos first) + negatives
    chosen = [pos] + negs
    labels = np.zeros((list_size,), dtype=np.float32)
    labels[0] = 1.0
    features = fe.extract_many(query, chosen).astype(np.float32, copy=False)
    return ListwiseSample(qid=qid, features=features, labels=labels)


def iter_listwise_samples(
    *,
    qids: Iterable[int],
    queries: dict[int, str],
    qrels: dict[int, set[str]],
    retriever: BM25Retriever,
    fe: FeatureExtractor,
    candidate_topn: int,
    list_size: int,
    seed: int,
    max_queries: Optional[int] = None,
) -> Iterator[ListwiseSample]:
    """
    Streams listwise samples by:
      qid -> BM25 top-N docs -> pick 1 pos + (list_size-1) negatives -> features+labels
    """
    n = 0
    for qid in qids:
        if max_queries is not None and n >= max_queries:
            break
        query = queries.get(qid)
        positives = qrels.get(qid)
        if query is None or not positives:
            continue
        docs = retriever.retrieve(query, topn=candidate_topn)
        sample = _sample_list_for_query(
            qid=qid,
            query=query,
            docs=docs,
            positives=positives,
            fe=fe,
            list_size=list_size,
            seed=seed,
        )
        if sample is None:
            continue
        yield sample
        n += 1


def iter_candidate_docs_from_cache(
    *,
    qids: Iterable[int],
    queries: dict[int, str],
    candidates: dict[int, list[CandidateDoc]],
    doc_lookup: callable,
) -> Iterator[tuple[int, str, list[Document]]]:
    """
    Placeholder for an optional future optimization:
    If you cache candidate docids and can map docid -> Document fields quickly,
    you can avoid re-running BM25 for each epoch.

    Current implementation uses BM25Retriever directly instead.
    """
    for qid in qids:
        q = queries.get(qid)
        if q is None:
            continue
        cands = candidates.get(qid, [])
        docs = [doc_lookup(c.docid, c.bm25) for c in cands]
        yield qid, q, docs




