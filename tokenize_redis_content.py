import argparse
from dotenv import load_dotenv
import os
import redis
import json
from tokenization import get_tokenizer
import multiprocessing as mp
from perf.simple_perf import perf_indicator

_WORKER_TOKENIZER = None


def _init_worker(_tok_backend_unused=None):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = get_tokenizer()


def _tokenize_doc(payload):
    key, body = payload
    tok = _WORKER_TOKENIZER or get_tokenizer()
    return key, tok.tokenize(body)

def connect_to_db(host: str, port: int):
    # Create a Redis client
    return redis.Redis(host=host, port=port, decode_responses=True)


@perf_indicator("tokenize_redis_content", "docs")
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Tokenize documents in Redis')
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
    )  # Default half of available cores
    args = parser.parse_args()

    redis_port = args.redis_port

    db = connect_to_db("localhost", redis_port)
    pipe = db.pipeline()
    # Track documents successfully tokenized and indexed
    num_docs_processed = 0
    local_tokenizer = get_tokenizer()
    # Create a multiprocessing pool
    pool = (
        mp.Pool(processes=args.workers, initializer=_init_worker)
        if args.workers > 1
        else None
    )
    print(f"[tokenize_redis_content] Using {args.workers} workers")

    # Iterate docs efficiently without blocking Redis
    def iter_doc_keys():
        for redis_key in db.scan_iter(match="D*", count=args.scan_count):
            if not isinstance(redis_key, str):
                try:
                    decoded_key = redis_key.decode()
                except Exception:
                    continue
            else:
                decoded_key = redis_key
            # Only yield keys that are not token keys
            if not decoded_key.startswith("token:"):
                yield decoded_key

    # Worker initializer creates a tokenizer in each process
    batch_keys = []
    for doc_key in iter_doc_keys():
        batch_keys.append(doc_key)
        if len(batch_keys) >= args.flush_every:
            # Fetch docs in batch
            document_contents = db.mget(batch_keys)
            docs = []
            for doc_key_single, content in zip(batch_keys, document_contents):
                try:
                    content_json = json.loads(content) if content else None
                except (json.JSONDecodeError, TypeError):
                    content_json = None
                if not content_json:
                    continue
                body_text = content_json.get("body", "")
                docs.append((doc_key_single, body_text))

            # Tokenize
            if pool and len(docs) > 1:
                tokenized = pool.map(_tokenize_doc, docs)
            else:
                tokenized = [
                    (doc_key_single, local_tokenizer.tokenize(body_text))
                    for doc_key_single, body_text in docs
                ]

            # Update document metric
            num_docs_processed += len(tokenized)

            # Aggregate postings by token (with or without positions)
            postings_by_token = {}
            for doc_key, tokens in tokenized:
                if args.store_tokens:
                    # update doc with tokens later
                    pass
                if args.store_positions:
                    # Build positions per token
                    positions_by_token = {}
                    for idx, tok in enumerate(tokens):
                        lst = positions_by_token.get(tok)
                        if lst is None:
                            positions_by_token[tok] = [idx]
                        else:
                            lst.append(idx)
                    # For each token found in this document, record its frequency and positions
                    for tok, positions in positions_by_token.items():
                        tf = len(positions)
                        per_doc_value = json.dumps({"tf": int(tf), "pos": positions})
                        mapping = postings_by_token.get(
                            tok
                        )  # get or create mapping of doc->posting for this token
                        if mapping is None:
                            postings_by_token[tok] = {doc_key: per_doc_value}
                        else:
                            mapping[doc_key] = per_doc_value
                else:
                    # Store only term frequencies
                    counts = {}
                    for tok in tokens:
                        counts[tok] = counts.get(tok, 0) + 1
                    for tok, tf in counts.items():
                        mapping = postings_by_token.get(tok)
                        if mapping is None:
                            postings_by_token[tok] = {
                                doc_key: json.dumps({"tf": int(tf)})
                            }
                        else:
                            mapping[doc_key] = json.dumps({"tf": int(tf)})

            # Write postings and optional doc updates
            if args.store_tokens:
                # if storing tokens, set the document values now
                # Reload current docs to update tokens to avoid holding big jsons
                contents = db.mget([k for k, _ in docs])
                for k, c, (_, tokens) in zip([k for k, _ in docs], contents, tokenized):
                    try:
                        content_json = json.loads(c) if c else None
                    except (json.JSONDecodeError, TypeError):
                        content_json = None
                    if not content_json:
                        continue
                    content_json["tokens"] = tokens
                    pipe.set(k, json.dumps(content_json))

            for tok, mapping in postings_by_token.items():
                token_key = f"token:{tok}"
                pipe.hset(token_key, mapping=mapping)

            pipe.execute()
            batch_keys = []

    # flush remainder
    if batch_keys:
        contents = db.mget(batch_keys)
        docs = []
        for doc_key_single, content in zip(batch_keys, contents):
            try:
                content_json = json.loads(content) if content else None
            except (json.JSONDecodeError, TypeError):
                content_json = None
            if not content_json:
                continue
            body_text = content_json.get("body", "")
            if not body_text:
                continue
            docs.append((doc_key_single, body_text))

        if pool and len(docs) > 1:
            tokenized = pool.map(_tokenize_doc, docs)
        else:
            tokenized = [
                (doc_key_single, local_tokenizer.tokenize(body_text))
                for doc_key_single, body_text in docs
            ]

        # Update document metric
        num_docs_processed += len(tokenized)

        postings_by_token = {}
        for doc_key_single, tokens in tokenized:
            if args.store_positions:
                # Build positions per token
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
                        postings_by_token[tok] = {doc_key_single: per_doc_value}
                    else:
                        mapping[doc_key_single] = per_doc_value
            else:
                counts = {}
                for tok in tokens:
                    counts[tok] = counts.get(tok, 0) + 1
                for tok, tf in counts.items():
                    mapping = postings_by_token.get(tok)
                    if mapping is None:
                        postings_by_token[tok] = {
                            doc_key_single: json.dumps({"tf": int(tf)})
                        }
                    else:
                        mapping[doc_key_single] = json.dumps({"tf": int(tf)})

        if args.store_tokens:
            contents = db.mget([doc_key_single for doc_key_single, _ in docs])
            for doc_key_single, content, (_, tokens) in zip(
                [doc_key_single for doc_key_single, _ in docs], contents, tokenized
            ):
                try:
                    content_json = json.loads(content) if content else None
                except (json.JSONDecodeError, TypeError):
                    content_json = None
                if not content_json:
                    continue
                content_json["tokens"] = tokens
                pipe.set(doc_key_single, json.dumps(content_json))

        for tok, mapping in postings_by_token.items():
            token_key = f"token:{tok}"
            pipe.hset(token_key, mapping=mapping)
        pipe.execute()

    # Cleanup multiprocessing pool
    if pool:
        pool.close()
        pool.join()

    print(f"[tokenize_redis_content] Documents processed: {num_docs_processed}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
