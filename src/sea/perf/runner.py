from typing import List
from sea.index.tokenization import get_tokenizer
from sea.ingest.pipeline import Ingestion, MinimalProcessor
from sea.perf.simple_perf import perf_indicator
from sea.query.search import search_documents
from sea.storage.interface import get_storage
from sea.utils.config import Config


def _read_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@perf_indicator("ingest", "docs")
def run_ingest(
    cfg: Config,
):
    db = get_storage(cfg=cfg)
    try:
        ingestion = Ingestion(db, [MinimalProcessor()], cfg.PERF_RUNNER.DOCUMENTS_PATH)
        inserted_keys = ingestion.ingest(cfg.PERF_RUNNER.BATCH_SIZE)
        # Cleanup newly inserted keys if requested
        if cfg.PERF_RUNNER.CLEANUP and inserted_keys:
            pipe = db.pipeline()
            for k in inserted_keys:
                pipe.delete(k)
            deleted = sum(int(bool(r)) for r in pipe.execute())
            print(f"[cleanup] deleted {deleted} keys")
        return None, cfg.PERF_RUNNER.BATCH_SIZE
    finally:
        db.close()


@perf_indicator("query", "queries")
def run_query(cfg: Config):
    queries = _read_queries(cfg.PERF_RUNNER.QUERIES_PATH)
    r = get_storage(cfg=cfg)
    tokenizer = get_tokenizer()

    for i in range(cfg.PERF_RUNNER.ITERATIONS):
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

    return None, cfg.PERF_RUNNER.ITERATIONS


def main():
    cfg = Config(load=True)

    if cfg.PERF_RUNNER.MODE == "ingest":
        run_ingest(cfg
        )
    elif cfg.PERF_RUNNER.MODE == "query":
        run_query(cfg)
    else:
        raise ValueError(f"Invalid mode: {cfg.PERF_RUNNER.MODE}")


if __name__ == "__main__":
    main()
