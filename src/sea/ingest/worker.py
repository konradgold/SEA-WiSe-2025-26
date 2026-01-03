# sea/ingest/worker.py
from __future__ import annotations
import array
from dataclasses import dataclass
from time import perf_counter
from typing import List, Tuple, Dict, Optional
import enum

from omegaconf import DictConfig

from sea.index.tokenization import get_tokenizer
from sea.storage.IO import BlockIO
from sea.utils.config_wrapper import Config

class Columns(enum.Enum):
    doc_id = "doc_id"
    url   = "url"
    title  = "title"
    body   = "body"

class Document:
    def __init__(self, doc_id: str, url: str, title: str, body: str):
        self.doc_id = doc_id
        self.url = url
        self.title = title
        self.body = body
    
    def set_running_id(self, running_id: int):
        self.running_id = running_id
        return self.doc_id
    
    def __getitem__(self, key: Optional[Columns]):
        if key == Columns.doc_id:
            return self.doc_id
        elif key == Columns.url:
            return self.url
        elif key == Columns.title:
            return self.title
        elif key == Columns.body:
            return self.body
        elif key is None:
            return f"{self.title} {self.body}"
        else:
            raise KeyError(f"Invalid column: {key}")

@dataclass
class BatchTimings:
    block_id: str
    n_docs: int
    build_index_s: float
    write_disk_s: float
    total_s: float

@dataclass
class BatchResult:
    metadata: Dict[int, list[str|int]]
    timings: BatchTimings 

# ---- Per-process singleton (created by init_worker) ----
_worker: "Worker | None" = None

class Worker:
    def __init__(self, store_positions: bool, cfg: Optional[DictConfig] = None,
                 field: Optional[str] = None) -> None:
        self.store_positions = store_positions
        self.blockIO = BlockIO(field=field)
        if cfg is None:
            cfg = Config(load=True)
        self.tokenizer = get_tokenizer(cfg)
        self.field = Columns(field) if field is not None else None

    # public entry point used by the parent to process one batch
    def process_batch(self, block_id: str, lines: List[Tuple[int, str]]) -> BatchResult:
        # build index and write shard on disk; return metadata for this batch
        t0 = perf_counter()
        print(f"[{block_id}] start")
        metadata, index = self._create_batch_index(lines)
        print(f"[{block_id}] index built")
        t1 = perf_counter()
        self.blockIO.write_block(block_id, index)
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

    def _create_batch_index(self, lines: List[Tuple[int, str]]):
        batch: list[Document] = []
        metadata: Dict[int, list[str|int]] = {}

        for idx, raw in lines:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            doc = Document(
                doc_id=parts[0],
                url=parts[1],
                title=parts[2],
                body=parts[3],
            )
            metadata[idx] = [doc.set_running_id(idx)]
              # replace doc_id with running index used downstream
            batch.append(doc)
        
        index = self._build_index(metadata, batch)
        return metadata, index

    def _build_index(self, metadata: Dict[int, list[str|int]], docs: list[Document]) -> dict[str, array.array[int]]:
        index: dict[str, array.array[int]] = {}
        for doc in docs:
            self._doc_to_postings(index, metadata, doc)
        return index

    def _doc_to_postings(self, index: dict[str, array.array[int]], metadata: Dict[int, list[str|int]], doc: Document):
        doc_id = doc.running_id # use the running index as doc_id
        tokens = self.tokenizer.tokenize(f"{doc[self.field]}")

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
            if self.store_positions:
                index[tok].extend(pos)

        metadata[doc_id].append(len(tokens))

# --------- top-level functions required by ProcessPoolExecutor ---------
def init_worker(field: Optional[str] = None):
    """
    Called once per worker process. Builds the heavy tokenizer and the Worker object.
    """
    global _worker

    cfg = Config(load=True)
    store_positions = cfg.TOKENIZER.STORE_POSITIONS
    _worker = Worker(store_positions=store_positions, cfg=cfg, field = field)

def process_batch(block_id: str, lines: List[Tuple[int, str]]) -> BatchResult:
    """
    Called for each batch from the parent. Uses the per-process _worker singleton.
    """
    assert _worker is not None, "Worker not initialized—did you pass initializer=init_worker?"
    return _worker.process_batch(block_id, lines)
