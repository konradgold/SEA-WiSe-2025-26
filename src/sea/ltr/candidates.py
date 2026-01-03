from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.msmarco import iter_msmarco_doc_qrels, iter_msmarco_doc_queries


@dataclass(frozen=True)
class CandidateDoc:
    docid: str
    bm25: float


def load_queries_map(path: str | Path) -> dict[int, str]:
    return {qid: q for qid, q in iter_msmarco_doc_queries(path)}


def load_qrels_map(path: str | Path) -> dict[int, set[str]]:
    qrels: dict[int, set[str]] = {}
    for q in iter_msmarco_doc_qrels(path):
        if q.rel <= 0:
            continue
        qrels.setdefault(q.qid, set()).add(q.docid)
    return qrels


def iter_qids(path: str | Path) -> Iterable[int]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield int(s)


def compute_candidate_recall(
    *,
    candidates: dict[int, list[CandidateDoc]],
    qrels: dict[int, set[str]],
) -> dict[str, float]:
    """
    Returns:
      - query_count: number of queries evaluated
      - recall_at_n: fraction of queries with >=1 positive in candidates
      - avg_pos_in_candidates: average number of positives retrieved per query (over queries with qrels)
    """
    q_eval = [qid for qid in candidates.keys() if qid in qrels]
    if not q_eval:
        return {"query_count": 0.0, "recall_at_n": 0.0, "avg_pos_in_candidates": 0.0}

    hit = 0
    pos_counts = 0
    for qid in q_eval:
        cand_ids = {c.docid for c in candidates[qid]}
        pos_ids = qrels[qid]
        num_pos_in = len(cand_ids & pos_ids)
        pos_counts += num_pos_in
        if num_pos_in > 0:
            hit += 1

    return {
        "query_count": float(len(q_eval)),
        "recall_at_n": hit / len(q_eval),
        "avg_pos_in_candidates": pos_counts / len(q_eval),
    }


def build_bm25_candidates(
    *,
    retriever: BM25Retriever,
    qids: Iterable[int],
    queries: dict[int, str],
    topn: int,
) -> dict[int, list[CandidateDoc]]:
    out: dict[int, list[CandidateDoc]] = {}
    for qid in qids:
        qtext = queries.get(qid)
        if qtext is None:
            continue
        docs = retriever.retrieve(qtext, topn=topn)
        out[qid] = [CandidateDoc(docid=d.doc_id, bm25=float(d.score)) for d in docs]
    return out


def write_candidates_jsonl(candidates: dict[int, list[CandidateDoc]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for qid in sorted(candidates.keys()):
            rec = {
                "qid": qid,
                "candidates": [{"docid": c.docid, "bm25": c.bm25} for c in candidates[qid]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_candidates_jsonl(path: str | Path) -> dict[int, list[CandidateDoc]]:
    p = Path(path)
    out: dict[int, list[CandidateDoc]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = int(rec["qid"])
            out[qid] = [CandidateDoc(**c) for c in rec["candidates"]]
    return out
