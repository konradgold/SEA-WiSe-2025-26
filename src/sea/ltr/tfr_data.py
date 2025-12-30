from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

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
    if list_size < 2:
        raise ValueError("list_size must be >= 2")

    id_map = {}
    for int_id, _ in id_results:
        orig_id, _ = retriever.ranker.storage_manager.getDocMetadataEntry(int_id)
        id_map[int_id] = orig_id

    pos_int_ids = [int_id for int_id, orig_id in id_map.items() if orig_id in positives]
    if not pos_int_ids:
        return None

    rng = random.Random((seed * 1_000_003) ^ qid)
    pos_id = rng.choice(pos_int_ids)

    neg_int_ids = [
        int_id for int_id, orig_id in id_map.items() if orig_id not in positives
    ]
    if not neg_int_ids:
        return None

    pool = neg_int_ids[: max(1, min(hard_pool_topk, len(neg_int_ids)))]
    num_neg = list_size - 1

    if len(pool) >= num_neg:
        negs = rng.sample(pool, num_neg)
    else:
        negs = [rng.choice(pool) for _ in range(num_neg)]

    # Build list of (id, score) pairs to hydrate
    # We shuffle the list so the positive document isn't always at index 0
    score_map = dict(id_results)
    chosen_pairs = [(pos_id, score_map[pos_id])] + [
        (nid, score_map[nid]) for nid in negs
    ]
    rng.shuffle(chosen_pairs)

    docs = retriever.hydrate_docs(chosen_pairs)
    if len(docs) != list_size:
        return None

    # 1.0 for the positive doc, 0.0 for others
    labels = np.array(
        [1.0 if p[0] == pos_id else 0.0 for p in chosen_pairs], dtype=np.float32
    )
    features = fe.extract_many(query, docs).astype(np.float32, copy=False)
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
    """
    Streams listwise samples by:
      qid -> BM25 top-N doc IDs (Fast) -> sample 1 pos + negatives -> hydrate docs (Slow, but minimized) -> features+labels
    """
    n = 0
    yielded = 0

    total_to_check = len(qids) if isinstance(qids, list) else None
    if max_queries is not None:
        total_to_check = (
            min(total_to_check, max_queries) if total_to_check else max_queries
        )

    pbar = tqdm.tqdm(qids, desc=description, total=total_to_check)

    for qid in pbar:
        if max_queries is not None and n >= max_queries:
            break
        query = queries.get(qid)
        positives = qrels.get(qid)
        if query is None or not positives:
            continue

        id_results = retriever.retrieve_ids(query, topn=candidate_topn)
        if not id_results:
            continue

        n += 1
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

        yielded += 1
        if n % 5 == 0:
            pbar.set_postfix({"hits": yielded, "recall": f"{yielded/n:.1%}"})
        yield sample


def iter_candidate_docs_from_cache(
    *,
    qids: Iterable[int],
    queries: dict[int, str],
    candidates: dict[int, list[CandidateDoc]],
    doc_lookup: callable,
) -> Iterator[tuple[int, str, list[Document]]]:
    for qid in qids:
        q = queries.get(qid)
        if q is None:
            continue
        cands = candidates.get(qid, [])
        docs = [doc_lookup(c.docid, c.bm25) for c in cands]
        yield qid, q, docs
