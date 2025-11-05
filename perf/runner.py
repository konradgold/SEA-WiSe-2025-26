import argparse
import os
from typing import List

from perf.simple_perf import perf_indicator

from ingestion import Ingestion, MinimalProcessor, connect_to_db
from main import connect_to_redis, search_documents
from tokenization import get_tokenizer

from utils.config import Config


def _read_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@perf_indicator("ingest", "docs")
def run_ingest(
    documents_path: str,
    batch_size: int,
    redis_host: str,
    redis_port: int,
    cleanup: bool = True,
):
    db = connect_to_db(redis_host, redis_port)
    try:
        ingestion = Ingestion(db, [MinimalProcessor()], documents_path)
        inserted_keys = ingestion.ingest(batch_size)
        # Cleanup newly inserted keys if requested
        if cleanup and inserted_keys:
            pipe = db.pipeline()
            for k in inserted_keys:
                pipe.delete(k)
            deleted = sum(int(bool(r)) for r in pipe.execute())
            print(f"[cleanup] deleted {deleted} keys")
        return None, batch_size
    finally:
        db.close()


@perf_indicator("query", "queries")
def run_query(queries_path: str, iterations: int, redis_host: str, redis_port: int):
    queries = _read_queries(queries_path)
    r = connect_to_redis(redis_host, redis_port)
    tokenizer = get_tokenizer()

    for i in range(iterations):
        q = queries[i % len(queries)]
        # tokenize
        _tokenize = perf_indicator("tokenize", "queries")(
            lambda x: (tokenizer.tokenize(x), 1)
        )
        tokens = _tokenize(q)
        # search
        _search = perf_indicator("search", "queries")(lambda t: (search_documents(r, t) or [], 1))
        results = _search(tokens)
        print(f"[query#{i+1}] q='{q}' tokens={len(tokens)} matches={len(results)}")

    return None, iterations


def main():
    cfg = Config(load=True)

    if cfg.PERF_RUNNER.MODE == "ingest":
        run_ingest(
            cfg.PERF_RUNNER.DOCUMENTS_PATH,
            cfg.PERF_RUNNER.BATCH_SIZE,
            cfg.REDIS_HOST,
            cfg.REDIS_PORT,
            cleanup=cfg.PERF_RUNNER.CLEANUP,
        )
    elif cfg.PERF_RUNNER.MODE == "query":
        run_query(cfg.PERF_RUNNER.QUERIES_PATH, cfg.PERF_RUNNER.ITERATIONS, cfg.REDIS_HOST, cfg.REDIS_PORT, cfg.TOKENIZER.BACKEND)
    else:
        raise ValueError(f"Invalid mode: {cfg.PERF_RUNNER.MODE}")


if __name__ == "__main__":
    main()
