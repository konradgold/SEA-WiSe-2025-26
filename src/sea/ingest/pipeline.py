from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from time import perf_counter
from types import SimpleNamespace
from typing import Dict, Iterator, List, Tuple

from sea.ingest.kmerger import KMerger
from sea.ingest.worker import  BatchTimings, init_worker, process_batch
import logging
from sea.storage.IO import DocDictonaryIO
from sea.utils.config_wrapper import Config
from sea.utils.logger import dir_size, write_message_to_log_file

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

    def ingest(self, num_documents: int = 1000, batch_size: int = 500):
        # TODO: adjust batch size based on L1 cache size of CPU
        if batch_size > num_documents or batch_size <= 0:
            batch_size = num_documents

        metadata = dict()

        cfg = Config(load=True)
        max_workers = cfg.TOKENIZER.NUM_WORKERS
        if cfg.TOKENIZER.NUM_WORKERS == 0:
            max_workers = cfg.TOKENIZER.NUM_WORKERS = (os.cpu_count() or 2) - 2
        # TODO build proper switch for different platforms
        mp_ctx = mp.get_context("spawn")   # "fork" on Linux/WSL; keep "spawn" on macOS/Windows
        counter = SimpleNamespace(value = 0)

        submitted = 0

        self._clear_existing_blocks()

        def submit_next(ex):
            nonlocal submitted
            if  counter.value >= num_documents:
                return None
            try:
                lines = next(batch_iter)
            except StopIteration:
                return None
            fut = ex.submit(process_batch, f"{submitted}", lines)
            submitted += 1
            print(f"Docs {counter.value:,} / {num_documents:,} submitted for processing.")
            return fut

        print(f"Starting ingestion of {num_documents} documents from {self.document_path}...")
        with open(self.document_path, "r") as f:
            batch_iter = self._chunked_lines(f, batch_size, counter)
            timings = []

            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_ctx,
                initializer=init_worker,
                initargs=(),
            ) as ex:

                # prefill the queue
                pending = set()
                submitted = 0

                for _ in range(max_workers):
                    fut =  submit_next(ex)
                    if fut is None:
                        break
                    pending.add(fut)

                while pending:
                    fut = next(as_completed(pending))
                    pending.remove(fut)

                    br = fut.result()
                    timings.append(br.timings)
                    metadata.update(br.metadata)

                    # try to submit a replacement
                    new_fut = submit_next(ex)
                    if new_fut is not None:
                        pending.add(new_fut)      

            print(f"Writing metadata for {len(metadata):,} documents to disk...")
            self._write_metadata_to_disk(metadata)
            print("Metadata writing complete.")
            return max_workers, timings

    def _write_metadata_to_disk(self, metadata: Dict[int, list[str]]):
        doc_dict_io = DocDictonaryIO(rewrite=True)
        doc_dict_io.write_metadata(metadata)
        doc_dict_io.close()

    def _clear_existing_blocks(self):
        cfg = Config(load=True)
        if os.path.exists(cfg.BLOCK_PATH):
            for f in os.listdir(cfg.BLOCK_PATH):
                os.remove(os.path.join(cfg.BLOCK_PATH, f))

    def _merge(self, a: dict, b: dict) -> None:
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


def summarize_pipeline(total_time, num_of_workers, batch_timings: List[BatchTimings], num_of_terms, merge_timing) -> None:
    total_docs = sum(bt.n_docs for bt in batch_timings)
    sum_build = sum(bt.build_index_s for bt in batch_timings) / num_of_workers
    sum_write = sum(bt.write_disk_s for bt in batch_timings) / num_of_workers

    ingestion_time = total_time - merge_timing

    # Weighted by docs: (sum step_time) / (sum docs)
    ms_per_doc_build = 1000.0 * (sum_build / total_docs) 
    ms_per_doc_write = 1000.0 * (sum_write / total_docs) 
    ms_per_doc_total = 1000.0 * (total_time / total_docs)

    # Throughput (docs/sec) across all workers
    build_docs_per_s = total_docs / sum_build 
    write_docs_per_s = total_docs / sum_write
    total_docs_per_s = total_docs / total_time
    
    # Measure disk usage at cfg.BLOCK_PATH
    cfg= Config(load=True)
    total_bytes_ = dir_size(cfg.BLOCK_PATH)

    time_measurement = (f"[RUN] workers={num_of_workers} docs={total_docs:,} terms={num_of_terms:,} "
            f"[TIME] time={total_time:.0f}s ingest={ingestion_time:.0f}s merge={merge_timing:.0f}s  "                        
            f"build={ms_per_doc_build:.2f} ms/doc ({build_docs_per_s:.0f} doc/s) "  
            f"write={ms_per_doc_write:.2f} ms/doc ({write_docs_per_s:.0f} doc/s) "
            f"total={ms_per_doc_total:.2f} ms/doc ({total_docs_per_s:.0f} doc/s) "
            f"[MEMORY] total={total_bytes_/1_048_576:.2f} MiB ({(total_bytes_/total_docs)/1_024:.2f} KiB/doc)")
    print(time_measurement)
    write_message_to_log_file(time_measurement)


def main():
    start = perf_counter()
    cfg = Config(load=True)
    ingestion = Ingestion(cfg.DOCUMENTS)
    no_of_workers, ingest_timings = ingestion.ingest(cfg.INGESTION.NUM_DOCUMENTS, cfg.INGESTION.BATCH_SIZE)

    merger = KMerger(cfg.BLOCK_PATH)
    no_of_terms, merge_timing = merger.merge_blocks()

    end = perf_counter()
    total_time = end - start

    summarize_pipeline(total_time, no_of_workers, ingest_timings, no_of_terms, merge_timing)


if __name__ == "__main__":
    main()
