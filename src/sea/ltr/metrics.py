from __future__ import annotations

import math
from typing import Iterable


def mrr_at_k(ranked_docids: list[str], relevant: set[str], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    for i, docid in enumerate(ranked_docids[:k], start=1):
        if docid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_docids: list[str], relevant: set[str], k: int = 10) -> float:
    """
    Binary-gain NDCG@k.
    """
    if k <= 0:
        return 0.0

    def dcg(ids: Iterable[str]) -> float:
        s = 0.0
        for i, docid in enumerate(list(ids)[:k], start=1):
            gain = 1.0 if docid in relevant else 0.0
            if gain > 0.0:
                s += gain / math.log2(i + 1)
        return s

    dcg_val = dcg(ranked_docids)
    # ideal DCG: all relevant docs first (binary gains)
    ideal_hits = min(k, len(relevant))
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0




