import argparse
import os
from typing import List

from perf.simple_perf import perf_indicator

from ingestion import Ingestion, MinimalProcessor, connect_to_db
from main import connect_to_redis, search_documents
from transformers import AutoTokenizer


def _read_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@perf_indicator("ingest", "docs")
def run_ingest(documents_path: str, batch_size: int, redis_host: str, redis_port: int):
    db = connect_to_db(redis_host, redis_port)
    try:
        ingestion = Ingestion(db, [MinimalProcessor()], documents_path)
        ingestion.ingest(batch_size)
        return None, batch_size
    finally:
        db.close()


@perf_indicator("query", "queries")
def run_query(queries_path: str, iterations: int, redis_host: str, redis_port: int, tokenizer_model: str):
    queries = _read_queries(queries_path)
    r = connect_to_redis(redis_host, redis_port)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    for i in range(iterations):
        q = queries[i % len(queries)]
        # tokenize
        _tokenize = perf_indicator("tokenize", "queries")(lambda x: (tokenizer.encode(x), 1))
        tokens = _tokenize(q)
        # search
        _search = perf_indicator("search", "queries")(lambda t: (search_documents(r, t) or [], 1))
        results = _search(tokens)
        print(f"[query#{i+1}] q='{q}' tokens={len(tokens)} matches={len(results)}")

    return None, iterations


def main():
    parser = argparse.ArgumentParser(description="Performance runner (simple console output)")
    parser.add_argument("--mode", choices=["ingest", "query"], required=True)
    parser.add_argument("--documents-path", type=str, default=os.getenv("DOCUMENTS", "msmarco-docs.tsv"))
    parser.add_argument("--queries-path", type=str, default="queries/sample_queries.txt")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--redis-host", type=str, default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", 6379)))
    parser.add_argument("--tokenizer-model", type=str, default=os.getenv("TOKENIZER_MODEL", "bert-base-cased"))

    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingest(args.documents_path, args.batch_size, args.redis_host, args.redis_port)
    elif args.mode == "query":
        run_query(args.queries_path, args.iterations, args.redis_host, args.redis_port, args.tokenizer_model)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()


