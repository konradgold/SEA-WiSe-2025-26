from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import tqdm

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
    id_results: list[tuple[int, float]],
    positives: set[str],
    retriever: BM25Retriever,
    fe: FeatureExtractor,
    list_size: int,
    seed: int,
    hard_pool_topk: int = 50,
) -> Optional[ListwiseSample]:
    """Sample one positive and hard negatives from BM25 candidates, then extract features."""
    if list_size < 2:
        raise ValueError("list_size must be >= 2")

    # Map internal IDs to original document IDs for qrels matching
    internal_to_original_id = {}
    for internal_id, _ in id_results:
        storage_manager = next(iter(retriever.ranker.storage_managers.values()))
        original_id, _ = storage_manager.getDocMetadataEntry(internal_id)
        internal_to_original_id[internal_id] = original_id

    positive_internal_ids = [internal_id for internal_id, original_id in internal_to_original_id.items() if original_id in positives]
    if not positive_internal_ids:
        return None

    random_generator = random.Random((seed * 1_000_003) ^ qid)
    selected_positive_id = random_generator.choice(positive_internal_ids)

    negative_internal_ids = [
        internal_id for internal_id, original_id in internal_to_original_id.items() if original_id not in positives
    ]
    if not negative_internal_ids:
        return None

    hard_negative_pool = negative_internal_ids[:hard_pool_topk] if negative_internal_ids else []
    num_negatives_needed = list_size - 1

    if len(hard_negative_pool) >= num_negatives_needed:
        selected_negative_ids = random_generator.sample(hard_negative_pool, num_negatives_needed)
    else:
        selected_negative_ids = [random_generator.choice(hard_negative_pool) for _ in range(num_negatives_needed)]

    # Build list of (id, score) pairs to hydrate and shuffle list with positive doc
    id_to_score = dict(id_results)
    selected_id_score_pairs = [(selected_positive_id, id_to_score[selected_positive_id])] + [
        (negative_id, id_to_score[negative_id]) for negative_id in selected_negative_ids
    ]
    random_generator.shuffle(selected_id_score_pairs)

    documents = retriever.hydrate_docs(selected_id_score_pairs)
    if len(documents) != list_size:
        return None

    # 1.0 for the positive doc, 0.0 for others
    labels = np.array(
        [1.0 if pair[0] == selected_positive_id else 0.0 for pair in selected_id_score_pairs], dtype=np.float32
    )
    features = fe.extract_many(query, documents).astype(np.float32, copy=False)
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
    description: str = "Generating samples",
) -> Iterator[ListwiseSample]:
    """Stream training samples

    Pipeline: query_id -> BM25 top-N doc IDs -> sample 1 positive + negatives
    -> hydrate docs from disk -> extract features + labels
    """
    queries_processed = 0
    samples_yielded = 0

    total_to_check = len(qids) if isinstance(qids, list) else None
    if max_queries is not None:
        total_to_check = (
            min(total_to_check, max_queries) if total_to_check else max_queries
        )

    progress_bar = tqdm.tqdm(qids, desc=description, total=total_to_check)

    for qid in progress_bar:
        if max_queries is not None and queries_processed >= max_queries:
            break
        query = queries.get(qid)
        positives = qrels.get(qid)
        if query is None or not positives:
            continue

        id_results = retriever.retrieve_ids(query, topn=candidate_topn)
        if not id_results:
            continue

        queries_processed += 1
        sample = _sample_list_for_query(
            qid=qid,
            query=query,
            id_results=id_results,
            positives=positives,
            retriever=retriever,
            fe=fe,
            list_size=list_size,
            seed=seed,
        )
        if sample is None:
            continue

        samples_yielded += 1
        if queries_processed % 5 == 0:
            progress_bar.set_postfix({"hits": samples_yielded, "recall": f"{samples_yielded/queries_processed:.1%}"})
        yield sample


def iter_candidate_docs_from_cache(
    *,
    qids: Iterable[int],
    queries: dict[int, str],
    candidates: dict[int, list[CandidateDoc]],
    doc_lookup: Callable,
) -> Iterator[tuple[int, str, list[Document]]]:
    """Iterate over queries and their cached candidate documents"""
    for qid in qids:
        query = queries.get(qid)
        if query is None:
            continue
        candidate_docs = candidates.get(qid, [])
        documents = [doc_lookup(candidate.docid, candidate.bm25) for candidate in candidate_docs]
        yield qid, query, documents
