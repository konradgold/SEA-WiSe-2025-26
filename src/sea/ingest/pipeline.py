import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import enum
import json
import os
from types import SimpleNamespace
from typing import Any, Iterator, List, Tuple

import psutil
from sea.index.tokenization import get_tokenizer
from sea.index.tokenizer_job import process_batch_in_memory
from sea.perf.simple_perf import perf_indicator
import logging
from sea.storage.interface import get_storage
from sea.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Columns(enum.Enum):
    doc_id = "doc_id"
    link = "link"
    title = "title"
    body = "body"

class Processor:
    def process(self, id, document) -> tuple[str, Any]:
        # Placeholder for processing logic
        return id, document

class MinimalProcessor(Processor):
    def process(self, id, document) -> tuple[str, str]:
        return document[0], json.dumps({
            Columns.doc_id.value: document[0],
            Columns.link.value: document[1],
            Columns.title.value: document[2],
            Columns.body.value: document[3],
        })

class Ingestion:
    """
    Ingest documents into the database
    Idea: Chain processors that are parsed via list
    """
    # def __init__(self, db, processors: list[Processor], document_path:str):
    def __init__(self, processors: list[Processor], document_path:str):

        # self.db = db
        self.processors = processors
        self.document_path = document_path


    def rss_mb(self, PROC):
        return PROC.memory_info().rss / (1024**2)

    @perf_indicator("ingest", "docs")
    def ingest(self, num_documents: int = 1000, batch_size: int = 500):
        # pipeline = self.db.pipeline()
        if batch_size > num_documents or batch_size <= 0:
            batch_size = num_documents


        MAX_MB = 1024
        PROC = psutil.Process(os.getpid())
        block_count = 0

        metadata = dict()
        merged: dict[str, dict[str, str]] = {}

        cfg = Config(load=True)
        if cfg.TOKENIZER.NUM_WORKERS == 0:
            cfg.TOKENIZER.NUM_WORKERS = (os.cpu_count() or 2) // 2
        print(f"[tokenize_redis_content] Using {cfg.TOKENIZER.NUM_WORKERS} worker{'s' if cfg.TOKENIZER.NUM_WORKERS != 1 else ''}")

        print(f"Starting ingestion of {num_documents} documents from {self.document_path}...")
        with open(self.document_path, "r") as f:
            ex = ProcessPoolExecutor(
                max_workers=cfg.TOKENIZER.NUM_WORKERS,
                mp_context=mp.get_context("spawn"))

            ex._processes  # touch to allocate
            ex.submit(lambda: None)  # noop to ensure pool up
            # index = 0
            index = SimpleNamespace(value = 0)
            while index.value < num_documents:
                futures = []
                for i, lines in enumerate(self._chunked_lines(f, batch_size, index)):
                    # if i * batch_size >= num_documents:
                        # break
                    futures.append(ex.submit(self._process_batch, lines))

                    if len(futures) == cfg.TOKENIZER.NUM_WORKERS:
                        break

                print(f"Docs {index.value} / {num_documents} submitted for processing.")

                for fut in as_completed(futures):
                    print(f"Doc batch processed.")
                    part_metadata, part = fut.result()
                    metadata.update(part_metadata)
                    self._merge(merged, part)

                print(f"Current RSS memory usage: {self.rss_mb(PROC):.2f} | {MAX_MB} MB")
                if index.value >= 10_000 or self.rss_mb(PROC) > MAX_MB:
                    print(f"Writing block {block_count} to disk, with {len(merged)} tokens and {len(metadata)} documents")

                    os.makedirs(cfg.BLOCK_PATH, exist_ok=True)
                    order_merged =  collections.OrderedDict(sorted(merged.items()))
                    with open(cfg.BLOCK_PATH + f"tokenizer_output_{block_count}.txt", "w") as block: 
                        for item in order_merged.items():
                            block.write(json.dumps(item) + "\n")

                    block_count += 1
                    merged.clear()
            
            if merged:
                print(f"Writing final block {block_count} to disk, with {len(merged)} tokens and {len(metadata)} documents")

                os.makedirs(cfg.BLOCK_PATH, exist_ok=True)
                order_merged =  collections.OrderedDict(sorted(merged.items()))
                with open(cfg.BLOCK_PATH + f"tokenizer_output_{block_count}.txt", "w") as block: 
                    for item in order_merged.items():
                        block.write(json.dumps(item) + "\n")
    
        #         keys_in_batch.append(doc_id)
        #         batch_count += 1
        #         total_visited += 1

        #         if batch_count >= batch_size:
        #             total_inserted = self._execute_pipe(pipeline, keys_in_batch, inserted_keys, total_inserted)
        #             remaining = num_documents - total_inserted
        #             batch_count = 0
        #             keys_in_batch = []
        #             percent = (total_inserted / num_documents * 100) if num_documents else 0.0
        #             logger.info(
        #                 f"Ingested {total_inserted:,}/{num_documents:,} ({percent:.1f}%) — "
        #                 f"visited {total_visited:,}; remaining {remaining:,}"
        #             )
        #             if total_inserted == 0 and total_visited % 1000 == 0:
        #                 logger.info("This is intended behavior if documents already exist in the database.")
        #         if remaining <= batch_count:
        #             break

        # if remaining > 0 and batch_count > 0:
        #     total_inserted = self._execute_pipe(pipeline, keys_in_batch, inserted_keys, total_inserted)

        # logger.info(f"There are now {self.db.dbsize()} documents in the database.")
        # Return (payload, count) for perf_indicator
        # return inserted_keys, total_inserted

    def _merge(self, a: dict, b: dict) -> None:
        # in-place merge: {tok: {doc: val}}
        for tok, mapping in b.items():
            a.setdefault(tok, {}).update(mapping)

    def _chunked_lines(self,f , batch_size: int,  counter :  SimpleNamespace) -> Iterator[List[Tuple[int, str]]]:
        buf = []
        for line in f:
            buf.append([counter.value , line])
            counter.value += 1

            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    
    def _process_batch(self, lines: List[Tuple[int, str]]) -> dict[str, dict[str, str]]:
        batch = []
        metadata = dict()

        for line in lines:
            index = line[0]
            data = line[1]
            doc = data.strip().split("\t")
            if len(doc) != 4:
                continue

            metadata[index] = [doc[0]] 
            doc[0] = index
            doc_id = doc[0]

            doc_id = doc[0]
            for processor in self.processors:
                doc_id, doc = processor.process(doc_id, doc)
            doc = json.loads(doc)

            batch.append(doc)

        index  = process_batch_in_memory(metadata, batch)
        return metadata, index

    def _execute_pipe(self, pipeline, keys_in_batch, inserted_keys, total_inserted):
        results = pipeline.execute()
        # Track which keys were newly inserted
        for k, r in zip(keys_in_batch, results):
            if bool(r):
                inserted_keys.append(k)
        ingested_now = sum(results)
        total_inserted += ingested_now
        return total_inserted

def main():
    cfg = Config(load=True)

    # cfg.STORAGE.KEEP_DOCUMENTS = True # otherwise this makes no sense
    # db = get_storage(cfg)
    ingestion = Ingestion([MinimalProcessor()], cfg.DOCUMENTS)
    # ingestion = Ingestion(db, [MinimalProcessor()], cfg.DOCUMENTS)
    ingestion.ingest(cfg.INGESTION.NUM_DOCUMENTS, cfg.INGESTION.BATCH_SIZE)
    # db.close()

if __name__ == "__main__":
    main()
