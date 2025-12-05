# sea/ingest/worker.py
from __future__ import annotations
import array
from collections import defaultdict, Counter
from dataclasses import dataclass
import struct
from time import perf_counter
from typing import Any, List, Tuple, Dict
import json
import os
import collections
import enum

from sea.index.tokenization import get_tokenizer
from sea.utils.config import Config  # only needed for _write_block_to_disk

class Columns(enum.Enum):
    doc_id = "doc_id"
    link   = "link"
    title  = "title"
    body   = "body"


@dataclass
class BatchTimings:
    block_id: str
    n_docs: int
    build_index_s: float
    write_disk_s: float
    total_s: float

@dataclass
class BatchResult:
    metadata: Dict[int, list[str]]
    timings: BatchTimings 

# ---- Per-process singleton (created by init_worker) ----
_worker: "Worker | None" = None

class Worker:
    def __init__(self, store_positions: bool, tokenizer):
        self.store_positions = store_positions
        self.tokenizer = tokenizer

    # public entry point used by the parent to process one batch
    def process_batch(self, block_id: str, lines: List[Tuple[int, str]]) -> Dict[int, list[str]]:
        # build index and write shard on disk; return metadata for this batch
        t0 = perf_counter()
        print(f"[{block_id}] start")
        metadata, index = self._create_batch_index(lines)
        print(f"[{block_id}] index built")
        t1 = perf_counter()
        self._write_block_to_disk(block_id, index)
        print(f"[{block_id}] written to disk")
        t2 = perf_counter()

        timings = BatchTimings(
            block_id=block_id,
            n_docs= len(lines),
            build_index_s=t1 - t0,
            write_disk_s=t2 - t1,
            total_s=t2 - t0,
        )
        return BatchResult(metadata=metadata, timings=timings)

    # ------------- internals -------------

    def _write_block_to_disk(self, block_id: str, index: Dict[str, array.array[int]]) -> None:
        cfg = Config(load=True)
        os.makedirs(cfg.BLOCK_PATH, exist_ok=True)

        path = os.path.join(cfg.BLOCK_PATH, f"tokenizer_output_{block_id}.bin")
        ordered = collections.OrderedDict(sorted(index.items()))

        with open(path, "wb") as out:
            out.write(b"SEAB\x01")  # magic + version 1
            for term, arr in ordered.items():
                # ensure uint32 array
                if arr.typecode != "I":
                    arr = array("I", arr)

                tb = term.encode("utf-8")
                out.write(struct.pack("<I", len(tb)))
                out.write(tb)

                out.write(struct.pack("<I", len(arr)))
                out.write(arr.tobytes())

    def _create_batch_index(self, lines: List[Tuple[int, str]]):
        batch: list[List[str]] = []
        metadata: Dict[int, list[str]] = {}

        for idx, raw in lines:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            original_id = parts[0]
            metadata[idx] = [original_id]
            parts[0] = idx  # replace doc_id with running index used downstream
            batch.append(parts)

        index = self._build_index(metadata, batch)
        return metadata, index

    def _build_index(self, metadata: Dict[int, list[str]], docs: list[list[str]]) -> dict[str, array.array[int]]:
        index: dict[str, array.array[int]] = {}
        for doc in docs:
            self._doc_to_postings(index, metadata, doc)
        return index

    def _doc_to_postings(self,index : dict[str, array.array[int]],  metadata: Dict[int, list[str]], doc: list[str]):
        doc_id = doc[0] # use the running index as doc_id
        # tokens = self.tokenizer.tokenize(f'{doc[2]} {doc[3]}')  # title + body
        tokens = doc[2].split() + doc[3].split()  # simple whitespace tokenizer

        if self.store_positions:
            pos_by_tok: dict[str, array.array[int]] = {}
            for i, tok in enumerate(tokens):
                if tok not in pos_by_tok:
                    pos_by_tok[tok] = array.array('I')  # unsigned int
                pos_by_tok[tok].append(i)
            for tok, pos in pos_by_tok.items():
                if tok not in index:
                    index[tok] = array.array('I')  # unsigned int
                index[tok].append(doc_id)
                index[tok].append(len(pos))
                index[tok].extend(pos)
        else:
            for tok, tf in Counter(tokens).items():
                index[tok].append(doc_id)
                index[tok].append(len(tf))

        metadata[doc_id].append(len(tokens))

# --------- top-level functions required by ProcessPoolExecutor ---------
def init_worker():
    """
    Called once per worker process. Builds the heavy tokenizer and the Worker object.
    """
    global _worker

    # tok = get_tokenizer()  # whatever your get_tokenizer expects (dict/dataclass)
    tok = None # simple whitespace tokenizer

    store_positions = True
    _worker = Worker(store_positions=store_positions, tokenizer=tok)

def process_batch(block_id: str, lines: List[Tuple[int, str]]):
    """
    Called for each batch from the parent. Uses the per-process _worker singleton.
    """
    assert _worker is not None, "Worker not initialized—did you pass initializer=init_worker?"
    return _worker.process_batch(block_id, lines)
