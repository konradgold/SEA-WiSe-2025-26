from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from sea.ltr.candidates import CandidateDoc


@dataclass(frozen=True)
class PairSample:
    qid: int
    pos_docid: str
    neg_docid: str
    pos_bm25: float
    neg_bm25: float


def sample_hard_negatives(
    *,
    qid: int,
    candidates: list[CandidateDoc],
    positives: set[str],
    neg_per_pos: int,
    seed: int,
    hard_pool_topk: int = 50,
) -> list[PairSample]:
    """
    Construct pairwise samples from BM25 candidates:
      - positives: judged positive docids (subset may or may not appear in candidates)
      - negatives: unjudged docs within the BM25 candidate list (excluding positives)

    Hard-negative heuristic:
      sample negatives preferentially from the highest-ranked BM25 docs (top-k pool).
    """
    rng = random.Random((seed * 1_000_003) ^ qid)

    cand_by_id = {c.docid: c for c in candidates}
    pos_in_cands = [pid for pid in positives if pid in cand_by_id]
    if not pos_in_cands:
        return []

    negs = [c for c in candidates if c.docid not in positives]
    if not negs:
        return []

    # Candidates are in BM25 rank order from retrieval, take top-k for hard negatives.
    pool = negs[: max(1, min(hard_pool_topk, len(negs)))]

    samples: list[PairSample] = []
    for pid in pos_in_cands:
        p = cand_by_id[pid]
        for _ in range(neg_per_pos):
            n = rng.choice(pool)
            samples.append(
                PairSample(
                    qid=qid,
                    pos_docid=p.docid,
                    neg_docid=n.docid,
                    pos_bm25=p.bm25,
                    neg_bm25=n.bm25,
                )
            )
    return samples


def iter_pair_samples(
    *,
    qids: Iterable[int],
    candidates: dict[int, list[CandidateDoc]],
    qrels: dict[int, set[str]],
    neg_per_pos: int,
    seed: int,
    hard_pool_topk: int = 50,
) -> Iterable[PairSample]:
    for qid in qids:
        if qid not in candidates or qid not in qrels:
            continue
        yield from sample_hard_negatives(
            qid=qid,
            candidates=candidates[qid],
            positives=qrels[qid],
            neg_per_pos=neg_per_pos,
            seed=seed,
            hard_pool_topk=hard_pool_topk,
        )
