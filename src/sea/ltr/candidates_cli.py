from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra

from sea.ltr.bm25 import BM25Retriever
from sea.ltr.candidates import (
    build_bm25_candidates,
    compute_candidate_recall,
    iter_qids,
    load_qrels_map,
    load_queries_map,
    write_candidates_jsonl,
)


@hydra.main(config_path="../../../configs", config_name="ltr_cli", version_base=None)
def main(cfg) -> None:
    queries = load_queries_map(cfg.queries)
    qrels = load_qrels_map(cfg.qrels)
    qids = list(iter_qids(cfg.qids))

    retriever = BM25Retriever.from_config()
    candidates = build_bm25_candidates(
        retriever=retriever, qids=qids, queries=queries, topn=cfg.topn
    )
    write_candidates_jsonl(candidates, cfg.out)

    if cfg.metrics_out:
        metrics = compute_candidate_recall(candidates=candidates, qrels=qrels)
        p = Path(cfg.metrics_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote diagnostics to {cfg.metrics_out}")


if __name__ == "__main__":
    main()
