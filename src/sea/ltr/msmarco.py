from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Qrel:
    qid: int
    docid: str
    rel: int = 1


def iter_msmarco_doc_queries(path: str | Path) -> Iterable[tuple[int, str]]:
    """
    MS MARCO doc train queries file in this repo is TSV:
        qid \\t query_text
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                # Skip malformed lines rather than crashing the whole pipeline.
                continue
            qid_s, query = parts
            try:
                qid = int(qid_s)
            except ValueError:
                continue
            yield (qid, query)


def iter_msmarco_doc_qrels(path: str | Path) -> Iterable[Qrel]:
    """
    MS MARCO qrels are typically in TREC format:
        qid 0 docid rel

    In this repo's `data/msmarco-doctrain-qrels.tsv`, the separator is whitespace.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid_s, _unused, docid, rel_s = parts[:4]
            try:
                qid = int(qid_s)
                rel = int(rel_s)
            except ValueError:
                continue
            yield Qrel(qid=qid, docid=docid, rel=rel)


@dataclass(frozen=True)
class QuerySplit:
    train: list[int]
    val: list[int]
    test: list[int]
    seed: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "train": self.train,
                "val": self.val,
                "test": self.test,
            },
            ensure_ascii=False,
            indent=2,
        )


def make_query_split(
    qids: Iterable[int],
    *,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> QuerySplit:
    qids_u = sorted(set(qids))
    if not qids_u:
        raise ValueError("No query ids provided for splitting")
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    rng = random.Random(seed)
    rng.shuffle(qids_u)

    n = len(qids_u)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # remainder to test for exact partition
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Split too small: n={n}, train={n_train}, val={n_val}, test={n_test}"
        )

    train = qids_u[:n_train]
    val = qids_u[n_train : n_train + n_val]
    test = qids_u[n_train + n_val :]

    # Sanity
    if len(set(train) & set(val)) or len(set(train) & set(test)) or len(set(val) & set(test)):
        raise AssertionError("Query split overlap detected")
    if len(train) + len(val) + len(test) != n:
        raise AssertionError("Query split does not partition all qids")

    return QuerySplit(train=train, val=val, test=test, seed=seed)


def persist_query_split(
    split: QuerySplit,
    *,
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_qids.txt").write_text("\n".join(map(str, split.train)) + "\n", encoding="utf-8")
    (out / "val_qids.txt").write_text("\n".join(map(str, split.val)) + "\n", encoding="utf-8")
    (out / "test_qids.txt").write_text("\n".join(map(str, split.test)) + "\n", encoding="utf-8")
    (out / "split_meta.json").write_text(split.to_json() + "\n", encoding="utf-8")




