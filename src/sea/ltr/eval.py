from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.candidates import iter_qids, load_qrels_map, load_queries_map
from sea.ltr.features import FeatureExtractor
from sea.ltr.metrics import mean, mrr_at_k, ndcg_at_k
from sea.utils.config import Config


def evaluate_split(
    *,
    qids: list[int],
    queries: dict[int, str],
    qrels: dict[int, set[str]],
    retriever: BM25Retriever,
    fe: FeatureExtractor,
    model,
    candidate_topn: int,
    k: int = 10,
    max_queries: Optional[int] = None,
) -> dict[str, float]:
    mrrs = []
    ndcgs = []
    used = 0

    for qid in qids:
        if max_queries is not None and used >= max_queries:
            break
        query = queries.get(qid)
        rel = qrels.get(qid)
        if query is None or not rel:
            continue

        docs = retriever.retrieve(query, topn=candidate_topn)
        if not docs:
            continue

        X = fe.extract_many(query, docs)
        scores = model.predict(X[None, :, :], verbose=0)[0]
        order = np.argsort(-scores)
        reranked = [docs[i].doc_id for i in order[:k]]

        mrrs.append(mrr_at_k(reranked, rel, k=k))
        ndcgs.append(ndcg_at_k(reranked, rel, k=k))
        used += 1

    return {
        "queries_evaluated": float(used),
        f"mrr@{k}": float(mean(mrrs)),
        f"ndcg@{k}": float(mean(ndcgs)),
    }


def evaluate_bm25_baseline(
    *,
    qids: list[int],
    queries: dict[int, str],
    qrels: dict[int, set[str]],
    retriever: BM25Retriever,
    candidate_topn: int,
    k: int = 10,
    max_queries: Optional[int] = None,
) -> dict[str, float]:
    mrrs = []
    ndcgs = []
    used = 0

    for qid in qids:
        if max_queries is not None and used >= max_queries:
            break
        query = queries.get(qid)
        rel = qrels.get(qid)
        if query is None or not rel:
            continue

        docs = retriever.retrieve(query, topn=candidate_topn)
        if not docs:
            continue
        ranked = [d.doc_id for d in docs[:k]]

        mrrs.append(mrr_at_k(ranked, rel, k=k))
        ndcgs.append(ndcg_at_k(ranked, rel, k=k))
        used += 1

    return {
        "queries_evaluated": float(used),
        f"mrr@{k}": float(mean(mrrs)),
        f"ndcg@{k}": float(mean(ndcgs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline evaluation: BM25 baseline vs TF-Ranking reranker.")
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--qrels", type=str, required=True)
    ap.add_argument("--split-dir", type=str, required=True)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--candidate-topn", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cfg = Config(load=True)
    queries = load_queries_map(args.queries)
    qrels = load_qrels_map(args.qrels)

    split_dir = Path(args.split_dir)
    qids_path = split_dir / f"{args.split}_qids.txt"
    qids = list(iter_qids(qids_path))

    retriever = BM25Retriever.from_config(cfg)
    fe = FeatureExtractor.from_config(cfg)

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model_path, compile=False)

    max_q = None if args.max_queries == 0 else int(args.max_queries)

    baseline = evaluate_bm25_baseline(
        qids=qids,
        queries=queries,
        qrels=qrels,
        retriever=retriever,
        candidate_topn=int(args.candidate_topn),
        k=int(args.k),
        max_queries=max_q,
    )
    reranker = evaluate_split(
        qids=qids,
        queries=queries,
        qrels=qrels,
        retriever=retriever,
        fe=fe,
        model=model,
        candidate_topn=int(args.candidate_topn),
        k=int(args.k),
        max_queries=max_q,
    )

    report = {
        "split": args.split,
        "candidate_topn": int(args.candidate_topn),
        "k": int(args.k),
        "bm25": baseline,
        "reranker": reranker,
    }
    print(json.dumps(report, indent=2))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
