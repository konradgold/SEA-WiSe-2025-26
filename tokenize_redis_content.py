import argparse
from dotenv import load_dotenv
import os
import redis
import json
from tokenization import get_tokenizer
from collections import Counter
import multiprocessing as mp

_WORKER_TOKENIZER = None


def _init_worker(_tok_backend_unused=None):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = get_tokenizer()


def _tokenize_doc(payload):
    key, body = payload
    tok = _WORKER_TOKENIZER or get_tokenizer()
    return key, tok.tokenize(body)

def connect_to_db(host: str, port: int):
    # Placeholder for database connection logic
    return redis.Redis(host=host, port=port, decode_responses=True)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Tokenize documents in Redis')
    parser.add_argument('--redis-port', type=int, default=int(os.getenv("REDIS_PORT", 6379)),
                      help='Redis server port')
    parser.add_argument('--flush-every', type=int, default=50,
                      help='number of documents per pipeline execute (default: 50)')
    parser.add_argument('--scan-count', type=int, default=1000,
                      help='hint for SCAN iteration batch size (default: 1000)')
    parser.add_argument('--store-tokens', action='store_true',
                      help='store token list inside each document (default: False)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                      help='number of parallel worker processes for tokenization')
    args = parser.parse_args()

    redis_port = args.redis_port

    db = connect_to_db("localhost", redis_port)
    pipe = db.pipeline()
    corr = 0
    # Iterate docs efficiently without blocking Redis
    def iter_doc_keys():
        for k in db.scan_iter(match='*', count=args.scan_count):
            if not isinstance(k, str):
                try:
                    kk = k.decode()
                except Exception:
                    continue
            else:
                kk = k
            if not kk.startswith('token:'):
                yield kk

    # Worker initializer creates a tokenizer in each process
    batch_keys = []
    for key in iter_doc_keys():
        batch_keys.append(key)
        if len(batch_keys) >= args.flush_every:
            # Fetch docs in batch
            contents = db.mget(batch_keys)
            docs = []
            for k, c in zip(batch_keys, contents):
                try:
                    content_json = json.loads(c) if c else None
                except (json.JSONDecodeError, TypeError):
                    content_json = None
                if not content_json:
                    continue
                body = content_json.get("body", "")
                docs.append((k, body))

            # Tokenize in parallel
            if args.workers > 1 and len(docs) > 1:
                with mp.Pool(processes=args.workers, initializer=_init_worker) as pool:
                    tokenized = pool.map(_tokenize_doc, docs)
            else:
                local_tok = get_tokenizer()
                tokenized = [(k, local_tok.tokenize(b)) for k, b in docs]

            # Aggregate postings by token
            postings_by_token = {}
            for doc_key, tokens in tokenized:
                if args.store_tokens:
                    # update doc with tokens
                    # re-fetch json from contents cache for simplicity
                    # Note: we could reuse earlier parsed jsons by mapping keys
                    pass
                counts = Counter(tokens)
                for tok, tf in counts.items():
                    mapping = postings_by_token.get(tok)
                    if mapping is None:
                        postings_by_token[tok] = {doc_key: int(tf)}
                    else:
                        mapping[doc_key] = int(tf)

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

            result = pipe.execute()
            corr += result.count(True)
            batch_keys = []

    # flush remainder
    if batch_keys:
        contents = db.mget(batch_keys)
        docs = []
        for k, c in zip(batch_keys, contents):
            try:
                content_json = json.loads(c) if c else None
            except (json.JSONDecodeError, TypeError):
                content_json = None
            if not content_json:
                continue
            body = content_json.get("body", "")
            docs.append((k, body))

        if args.workers > 1 and len(docs) > 1:
            with mp.Pool(processes=args.workers, initializer=_init_worker) as pool:
                tokenized = pool.map(_tokenize_doc, docs)
        else:
            local_tok = get_tokenizer()
            tokenized = [(k, local_tok.tokenize(b)) for k, b in docs]

        postings_by_token = {}
        for doc_key, tokens in tokenized:
            counts = Counter(tokens)
            for tok, tf in counts.items():
                mapping = postings_by_token.get(tok)
                if mapping is None:
                    postings_by_token[tok] = {doc_key: int(tf)}
                else:
                    mapping[doc_key] = int(tf)

        if args.store_tokens:
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

        result = pipe.execute()
        corr += result.count(True)
    print(f"Number of correct tokenizations: {corr}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
