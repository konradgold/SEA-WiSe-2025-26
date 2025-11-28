import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from types import SimpleNamespace
from typing import Any, Iterator, List, Tuple

from sea.ingest.worker import BatchResult, BatchTimings, init_worker, process_batch
from sea.perf.simple_perf import perf_indicator
import logging
from sea.utils.config import Config
from sea.utils.logger import write_message_to_log_file

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
            timings = []
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
                        batch_result = fut.result()
                        timings.append(batch_result.timings)
                        metadata.update(batch_result.metadata)                    
                    iteration += 1
            self.summarize(timings)
            return [], counter.value

    def summarize(self, batch_timings: List[BatchTimings]) -> None:
        total_docs = sum(bt.n_docs for bt in batch_timings)
        sum_build = sum(bt.build_index_s for bt in batch_timings)
        sum_write = sum(bt.write_disk_s for bt in batch_timings)
        sum_total = sum(bt.total_s for bt in batch_timings)

        # Weighted by docs: (sum step_time) / (sum docs)
        ms_per_doc_build = 1000.0 * (sum_build / total_docs) if total_docs else 0.0
        ms_per_doc_write = 1000.0 * (sum_write / total_docs) if total_docs else 0.0
        ms_per_doc_total = 1000.0 * (sum_total / total_docs) if total_docs else 0.0

        # Throughput (docs/sec) across all workers
        build_docs_per_s = total_docs / sum_build if sum_build else float("inf")
        write_docs_per_s = total_docs / sum_write if sum_write else float("inf")
        total_docs_per_s = total_docs / sum_total if sum_total else float("inf")
        
        time_measurement = (f"[AGG] docs={total_docs}  build={ms_per_doc_build:.2f} ms/doc "
                f"({build_docs_per_s:.0f} doc/s)  write={ms_per_doc_write:.2f} ms/doc "
                f"({write_docs_per_s:.0f} doc/s)  total={ms_per_doc_total:.2f} ms/doc "
                f"({total_docs_per_s:.0f} doc/s)")
        print(time_measurement)
        write_message_to_log_file(time_measurement)

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
