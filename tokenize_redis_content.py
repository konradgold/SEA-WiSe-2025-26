import argparse
from dotenv import load_dotenv
import os
import redis
import json
from tokenization import get_tokenizer
import multiprocessing as mp
from perf.simple_perf import perf_indicator
from utils.config import Config
from collections import Counter

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


def process_batch(db, pipe, batch_keys, cfg, pool, local_tokenizer):
    docs = load_documents(db, batch_keys)
    if not docs:
        return 0

    tokenized = tokenize_documents(docs, pool, local_tokenizer)
    postings_by_token = build_postings(tokenized, cfg.TOKENIZER.STORE_POSITIONS)

    if args.TOKENIZER.STORE_TOKENS:
        update_documents_with_tokens(db, docs, tokenized, pipe)

    write_postings(postings_by_token, pipe)
    pipe.execute()
    return len(tokenized)



def connect_to_db(host: str, port: int, passwort=None):
    return redis.Redis(host=host, port=port, passwort=passwort, decode_responses=True)


@perf_indicator("tokenize_docs", "docs")
def main():
    load_dotenv()
    cfg = Config(load=True)
    if cfg.TOKENIZER.WORKERS == 0:
      cfg.TOKENIZER.WORKERS = (os.cpu_count() or 2) // 2
      

    db = connect_to_db(cfg.REDIS_HOST, cfg.REDIS_PORT)
    pipe = db.pipeline()
    num_docs_processed = 0
    local_tokenizer = get_tokenizer()
    pool = (
        mp.Pool(processes=cfg.TOKENIZER.WORKERS, initializer=_init_worker)
        if cfg.TOKENIZER.WORKERS > 1
        else None
    )
    print(f"[tokenize_redis_content] Using {cfg.TOKENIZER.WORKERS} workers")

    batch_keys = []
    for doc_key in iter_doc_keys(db, cfg.TOKENIZER.SCAN_COUNT):
        batch_keys.append(doc_key)
        if len(batch_keys) >= cfg.TOKENIZER.FLUSH_EVERY:
            num_docs_processed += process_batch(
                db, pipe, batch_keys, cfg, pool, local_tokenizer
            )
            batch_keys = []

    if batch_keys:
        num_docs_processed += process_batch(
            db, pipe, batch_keys, cfg, pool, local_tokenizer
        )
    if pool:
        pool.close()
        pool.join()

    print(f"[tokenize_redis_content] Documents processed: {num_docs_processed}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
