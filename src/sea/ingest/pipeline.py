import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from types import SimpleNamespace
from typing import Any, Iterator, List, Tuple

from sea.ingest.worker import init_worker, process_batch
from sea.perf.simple_perf import perf_indicator
import logging
from sea.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class Ingestion:
    """
    Ingest documents into the database
    Idea: Chain processors that are parsed via list
    """
    # def __init__(self, db, processors: list[Processor], document_path:str):
    def __init__(self, document_path:str):

        self.document_path = document_path


    def rss_mb(self, PROC):
        return PROC.memory_info().rss / (1024**2)

    @perf_indicator("ingest", "docs")
    def ingest(self, num_documents: int = 1000, batch_size: int = 500):
        if batch_size > num_documents or batch_size <= 0:
            batch_size = num_documents

        metadata = dict()

        cfg = Config(load=True)
        max_workers = cfg.TOKENIZER.NUM_WORKERS
        if cfg.TOKENIZER.NUM_WORKERS == 0:
            max_workers = cfg.TOKENIZER.NUM_WORKERS = (os.cpu_count() or 2) - 2
        #TODO build proper switch for different platforms
        mp_ctx = mp.get_context("spawn")   # "fork" on Linux/WSL; keep "spawn" on macOS/Windows
        counter = SimpleNamespace(value = 0)

        # print(f"[tokenize_redis_content] Using {cfg.TOKENIZER.NUM_WORKERS} worker{'s' if cfg.TOKENIZER.NUM_WORKERS != 1 else ''}")

        print(f"Starting ingestion of {num_documents} documents from {self.document_path}...")


        with open(self.document_path, "r") as f:
            batch_iter = self._chunked_lines(f, batch_size, counter)
            iteration = 0

            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_ctx,
                initializer=init_worker,
                initargs=(),
            ) as ex:


                while counter.value < num_documents:
                    futures = []
                    worker_num = 0  
                    for _ in range(max_workers):
                        try:
                            lines = next(batch_iter)
                        except StopIteration:
                            break
                        futures.append(ex.submit(process_batch, f"{iteration}-{worker_num}", lines))
                        worker_num += 1

                    if not futures:   # nothing left to process; exit outer loop
                        break

                    print(f"Docs {counter.value} / {num_documents} submitted for processing.")

                    for fut in as_completed(futures):
                        part_metadata = fut.result()
                        metadata.update(part_metadata)
                    iteration += 1
            return [], counter.value



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
    ingestion = Ingestion(cfg.DOCUMENTS)
    # ingestion = Ingestion(db, [MinimalProcessor()], cfg.DOCUMENTS)
    ingestion.ingest(cfg.INGESTION.NUM_DOCUMENTS, cfg.INGESTION.BATCH_SIZE)
    # db.close()

if __name__ == "__main__":
    main()
