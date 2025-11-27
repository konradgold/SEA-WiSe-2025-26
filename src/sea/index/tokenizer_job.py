from functools import partial
from typing import List
from dotenv import load_dotenv
import os
import json
import logging
import multiprocessing as mp
from sea.index.tokenization import get_tokenizer
from sea.perf.simple_perf import perf_indicator
from sea.storage.interface import LocalStorage, get_storage
from sea.utils.config import Config
from collections import Counter, defaultdict


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
        (doc["doc_id"], local_tokenizer.tokenize(doc["title"] + " " + doc["body"])) for doc in docs
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
        pipe.hset(token_key, mapping)


def process_batch(db, pipe, batch_keys, cfg, pool, local_tokenizer):
    docs = load_documents(db, batch_keys)
    if not docs:
        return 0

    tokenized = tokenize_documents(docs, pool, local_tokenizer)
    postings_by_token = build_postings(tokenized, cfg.TOKENIZER.STORE_POSITIONS)

    if cfg.TOKENIZER.STORE_TOKENS:
        update_documents_with_tokens(db, docs, tokenized, pipe)

    write_postings(postings_by_token, pipe)
    pipe.execute()
    return len(tokenized)


def doc_to_postings(doc: dict, tokenizer, store_positions: bool) -> dict[str, dict[str, str]]:
    """
    Returns postings for ONE doc:
      { token: { doc_id: json_value } }
    """
    doc_id = doc["doc_id"]
    tokens = tokenizer.tokenize(f'{doc.get("title","")} {doc.get("body","")}')
    result: dict[str, dict[str, str]] = {}

    if store_positions:
        pos_by_tok: dict[str, list[int]] = defaultdict(list)
        for idx, tok in enumerate(tokens):
            pos_by_tok[tok].append(idx)
        for tok, positions in pos_by_tok.items():
            value = json.dumps({"tf": len(positions), "pos": positions})
            result[tok] = {doc_id: value}
    else:
        for tok, tf in Counter(tokens).items():
            value = json.dumps({"tf": int(tf)})
            result[tok] = {doc_id: value}

    return result, len(tokens)

def build_index(metadata, docs: list[dict], tokenizer, store_positions: bool):
    postings_by_token: dict[str, dict[str, str]] = defaultdict(dict)

    for doc in docs:
        part, token_no = doc_to_postings(doc, tokenizer, store_positions)
        metadata[doc["doc_id"]].append(token_no)    
        for tok, mapping in part.items():
            postings_by_token[tok].update(mapping)

    return dict(postings_by_token)



def process_batch_in_memory(metadata, docs : List[dict] = []) -> dict:
    if len(docs) == 0:
        return dict()
    
    load_dotenv()
    cfg = Config(load=True)
    local_tokenizer = get_tokenizer(cfg)
    return  build_index(metadata, docs, local_tokenizer, cfg.TOKENIZER.STORE_POSITIONS)


@perf_indicator("tokenize_redis_content", "docs")
def main():
    load_dotenv()
    cfg = Config(load=True)
    if cfg.TOKENIZER.NUM_WORKERS == 0:
      cfg.TOKENIZER.NUM_WORKERS = (os.cpu_count() or 2) // 2
      

    db = get_storage(cfg)
    pipe = db.pipeline()
    num_docs_processed = 0
    local_tokenizer = get_tokenizer(cfg)
    pool = (
        mp.Pool(processes=cfg.TOKENIZER.NUM_WORKERS, initializer=_init_worker)
        if cfg.TOKENIZER.NUM_WORKERS > 1
        else None
    )
    print(f"[tokenize_redis_content] Using {cfg.TOKENIZER.NUM_WORKERS} worker{'s' if cfg.TOKENIZER.NUM_WORKERS != 1 else ''}")

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
    
    if isinstance(db, LocalStorage):
        db.save()

    print(f"[tokenize_redis_content] Documents processed: {num_docs_processed}")
    return None, num_docs_processed



if __name__ == "__main__":
    mp.freeze_support()
    main()
