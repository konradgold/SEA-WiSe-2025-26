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

    # Find which of the top-N IDs are in the positive set
    # MS MARCO docids in qrels are strings (e.g. 'D1234'),
    # but the ranker returns internal integer IDs.
    # We need to map internal IDs to original IDs to check against positives.

    # Efficiently find all docids for the retrieved set
    id_map = {}  # internal_id -> original_id
    for int_id, score in id_results:
        orig_id, _len = retriever.ranker.storage_manager.getDocMetadataEntry(int_id)
        id_map[int_id] = orig_id

    pos_int_ids = [int_id for int_id, orig_id in id_map.items() if orig_id in positives]
    if not pos_int_ids:
        return None

    # Pick one positive (deterministic-ish but seedable)
    rng = random.Random((seed * 1_000_003) ^ qid)
    pos_id = rng.choice(pos_int_ids)

    neg_int_ids = [
        int_id for int_id, orig_id in id_map.items() if orig_id not in positives
    ]
    if not neg_int_ids:
        return None

    pool = neg_int_ids[: max(1, min(hard_pool_topk, len(neg_int_ids)))]
    num_neg = list_size - 1
    negs = [rng.choice(pool) for _ in range(num_neg)]

    # Build list of (id, score) pairs to hydrate
    # We retrieve the score from id_results for each chosen ID
    score_map = dict(id_results)
    chosen_pairs = [(pos_id, score_map[pos_id])] + [
        (nid, score_map[nid]) for nid in negs
    ]

    # Hydrate ONLY the chosen docs from disk
    docs = retriever.hydrate_docs(chosen_pairs)
    if len(docs) != list_size:
        return None

    labels = np.zeros((list_size,), dtype=np.float32)
    labels[0] = 1.0
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

    # Wrap with tqdm for progress feedback
    pbar = tqdm.tqdm(qids, desc=description, total=max_queries if max_queries else None)

    for qid in pbar:
        if max_queries is not None and n >= max_queries:
            break
        query = queries.get(qid)
        positives = qrels.get(qid)
        if query is None or not positives:
            continue

        # 1. Get ONLY IDs first (Fast)
        id_results = retriever.retrieve_ids(query, topn=candidate_topn)
        if not id_results:
            continue

        n += 1

        # 2. Sample and hydrate (Minimize disk I/O)
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

    if yielded == 0:
        print(f"\nWARNING: Zero samples were generated from {n} queries.")
        print(
            "This usually means the positive document for these queries was not in the top-N BM25 results."
        )
        print(
            "Check if you have ingested the full MS MARCO dataset or if candidate_topn is too low."
        )


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
