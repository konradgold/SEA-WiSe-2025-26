from __future__ import annotations

import argparse
import json
from pathlib import Path

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.candidates import (
    build_bm25_candidates,
    compute_candidate_recall,
    iter_qids,
    load_qrels_map,
    load_queries_map,
    write_candidates_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute BM25 top-N candidates for a set of query ids.")
    ap.add_argument("--queries", type=str, required=True, help="Path to queries TSV (qid\\ttext).")
    ap.add_argument("--qrels", type=str, required=True, help="Path to qrels file (qid 0 docid rel).")
    ap.add_argument("--qids", type=str, required=True, help="Path to file with one qid per line.")
    ap.add_argument("--topn", type=int, default=200)
    ap.add_argument("--out", type=str, required=True, help="Output JSONL file for candidates.")
    ap.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional path to write candidate recall diagnostics as JSON.",
    )
    args = ap.parse_args()

    queries = load_queries_map(args.queries)
    qrels = load_qrels_map(args.qrels)
    qids = list(iter_qids(args.qids))

    retriever = BM25Retriever.from_config()
    candidates = build_bm25_candidates(
        retriever=retriever, qids=qids, queries=queries, topn=args.topn
    )
    write_candidates_jsonl(candidates, args.out)
    print(f"Wrote candidates to {args.out} (queries={len(candidates)}, topn={args.topn})")

    metrics = compute_candidate_recall(candidates=candidates, qrels=qrels)
    print(f"Candidate recall@{args.topn}: {metrics}")
    if args.metrics_out:
        p = Path(args.metrics_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote diagnostics to {args.metrics_out}")


if __name__ == "__main__":
    main()


