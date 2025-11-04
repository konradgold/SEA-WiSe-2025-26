import argparse
from dotenv import load_dotenv
import os
import redis
import json
import logging
from tokenization import get_tokenizer
import multiprocessing as mp
from perf.simple_perf import perf_indicator
from collections import Counter


logging.basicConfig(level=logging.INFO)
# Derive logger name from filename
_module_name = os.path.splitext(os.path.basename(__file__))[0]
logger = logging.getLogger(_module_name)

_WORKER_TOKENIZER = None


def _init_worker():
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = get_tokenizer()


def _tokenize_doc(payload):
    key, body = payload
    tok = _WORKER_TOKENIZER or get_tokenizer()
    return key, tok.tokenize(body)


def iter_doc_keys(db, scan_count):
    for redis_key in db.scan_iter(match="D*", count=scan_count):
        if not isinstance(redis_key, str):
            try:
                decoded_key = redis_key.decode()
            except Exception:
                continue
        else:
            decoded_key = redis_key
        if not decoded_key.startswith("token:"):
            yield decoded_key


def load_documents(db, keys):
    contents = db.mget(keys)
    docs = []
    for key, content in zip(keys, contents):
        try:
            content_json = json.loads(content) if content else None
        except (json.JSONDecodeError, TypeError):
            content_json = None
        if not content_json:
            continue
        body_text = content_json.get("body", "")
        if body_text:
            docs.append((key, body_text))
    return docs


def tokenize_documents(docs, pool, local_tokenizer):
    if pool and len(docs) > 1:
        return pool.map(_tokenize_doc, docs)
    return [
        (doc_key, local_tokenizer.tokenize(body_text)) for doc_key, body_text in docs
    ]


def build_postings(tokenized, store_positions: bool):
    postings_by_token = {}
    for doc_key, tokens in tokenized:
        if store_positions:
            positions_by_token = {}
            for idx, tok in enumerate(tokens):
                lst = positions_by_token.get(tok)
                if lst is None:
                    positions_by_token[tok] = [idx]
                else:
                    lst.append(idx)
            for tok, positions in positions_by_token.items():
                tf = len(positions)
                per_doc_value = json.dumps({"tf": int(tf), "pos": positions})
                mapping = postings_by_token.get(tok)
                if mapping is None:
                    postings_by_token[tok] = {doc_key: per_doc_value}
                else:
                    mapping[doc_key] = per_doc_value
        else:
            counts = Counter(tokens)
            for tok, tf in counts.items():
                mapping = postings_by_token.get(tok)
                value = json.dumps({"tf": int(tf)})
                if mapping is None:
                    postings_by_token[tok] = {doc_key: value}
                else:
                    mapping[doc_key] = value
    return postings_by_token


def update_documents_with_tokens(db, docs, tokenized, pipe):
    doc_keys = [k for k, _ in docs]
    contents = db.mget(doc_keys)
    for k, c, (_, tokens) in zip(doc_keys, contents, tokenized):
        try:
            content_json = json.loads(c) if c else None
        except (json.JSONDecodeError, TypeError):
            content_json = None
        if not content_json:
            continue
        content_json["tokens"] = tokens
        pipe.set(k, json.dumps(content_json))


def write_postings(postings_by_token, pipe):
    for tok, mapping in postings_by_token.items():
        token_key = f"token:{tok}"
        pipe.hset(token_key, mapping=mapping)


def process_batch(db, pipe, batch_keys, args, pool, local_tokenizer):
    docs = load_documents(db, batch_keys)
    if not docs:
        return 0

    tokenized = tokenize_documents(docs, pool, local_tokenizer)
    postings_by_token = build_postings(tokenized, args.store_positions)

    if args.store_tokens:
        update_documents_with_tokens(db, docs, tokenized, pipe)

    write_postings(postings_by_token, pipe)
    pipe.execute()
    return len(tokenized)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tokenize documents in Redis")
    parser.add_argument(
        "--redis-port",
        type=int,
        default=int(os.getenv("REDIS_PORT", 6379)),
        help="Redis server port",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="number of documents per pipeline execute (default: 50)",
    )
    parser.add_argument(
        "--scan-count",
        type=int,
        default=1000,
        help="hint for SCAN iteration batch size (default: 1000)",
    )
    parser.add_argument(
        "--store-tokens",
        action="store_true",
        help="store token list inside each document (default: False)",
    )
    # Toggle storing positional indices in postings (default: True)
    parser.add_argument(
        "--store-positions",
        dest="store_positions",
        action="store_true",
        help="store positional indices per token (default: True)",
    )
    parser.add_argument(
        "--no-store-positions",
        dest="store_positions",
        action="store_false",
        help="do not store positions; store only term frequencies",
    )
    parser.set_defaults(store_positions=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="number of parallel worker processes for tokenization",
    )
    return parser.parse_args()

def connect_to_db(host: str, port: int):
    return redis.Redis(host=host, port=port, decode_responses=True)


@perf_indicator("tokenize_redis_content", "docs")
def main():
    load_dotenv()
    args = parse_arguments()

    db = connect_to_db("localhost", args.redis_port)
    pipe = db.pipeline()
    num_docs_processed = 0
    local_tokenizer = get_tokenizer()
    pool = (
        mp.Pool(processes=args.workers, initializer=_init_worker)
        if args.workers > 1
        else None
    )
    logger.info(f"Using {args.workers} workers")

    batch_keys = []
    for doc_key in iter_doc_keys(db, args.scan_count):
        batch_keys.append(doc_key)
        if len(batch_keys) >= args.flush_every:
            num_docs_processed += process_batch(
                db, pipe, batch_keys, args, pool, local_tokenizer
            )
            batch_keys = []

    if batch_keys:
        num_docs_processed += process_batch(
            db, pipe, batch_keys, args, pool, local_tokenizer
        )
    if pool:
        pool.close()
        pool.join()

    return None, num_docs_processed


if __name__ == "__main__":
    mp.freeze_support()
    main()
