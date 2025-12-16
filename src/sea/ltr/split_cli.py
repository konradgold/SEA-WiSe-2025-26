from __future__ import annotations

import argparse
from pathlib import Path

from sea.ltr.msmarco import iter_msmarco_doc_qrels, make_query_split, persist_query_split


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a deterministic MS MARCO query-id split.")
    ap.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to MS MARCO doc qrels file (e.g., data/msmarco-doctrain-qrels.tsv).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for split files (e.g., data/splits).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qids = (q.qid for q in iter_msmarco_doc_qrels(args.qrels) if q.rel > 0)
    split = make_query_split(qids, seed=args.seed)
    persist_query_split(split, out_dir=Path(args.out_dir))

    print(
        f"Wrote split to {args.out_dir} "
        f"(train={len(split.train)}, val={len(split.val)}, test={len(split.test)}, seed={split.seed})"
    )


if __name__ == "__main__":
    main()




