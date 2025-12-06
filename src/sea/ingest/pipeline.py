from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq
import multiprocessing as mp
import os
import struct
from time import perf_counter
from types import SimpleNamespace
from typing import Iterable, Iterator, List, Tuple

from sea.ingest.worker import  BatchTimings, init_worker, process_batch
import logging
from sea.utils.config import Config
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
        #TODO: adjust batch size based on L1 cache size of CPU
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
            print(f"Docs {counter.value} / {num_documents} submitted for processing.")
            return fut

        print(f"Starting ingestion of {num_documents} documents from {self.document_path}...")
        start = perf_counter()

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
                    # fut = ex.submit(process_batch, f"{submitted}", lines)
                    pending.add(fut)
                    # submitted += 1
                
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
            end = perf_counter()
            self.summarize(end - start,max_workers ,timings)
            return [], counter.value

    def summarize(self, total_time, num_of_workers, batch_timings: List[BatchTimings]) -> None:
        total_docs = sum(bt.n_docs for bt in batch_timings)
        sum_build = sum(bt.build_index_s for bt in batch_timings) / num_of_workers
        sum_write = sum(bt.write_disk_s for bt in batch_timings) / num_of_workers

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

        time_measurement = (f"[AGG] workers={num_of_workers} docs={total_docs} time={total_time:.0f}s "
                f"build={ms_per_doc_build:.2f} ms/doc ({build_docs_per_s:.0f} doc/s) "  
                f"write={ms_per_doc_write:.2f} ms/doc ({write_docs_per_s:.0f} doc/s) "
                f"total={ms_per_doc_total:.2f} ms/doc ({total_docs_per_s:.0f} doc/s) "
                f"total={total_bytes_/1_048_576:.2f} MiB ({(total_bytes_/total_docs)/1_024:.2f} KiB/doc)")
        print(time_measurement)
        write_message_to_log_file(time_measurement)
    
    def _clear_existing_blocks(self):
        cfg = Config(load=True)
        if os.path.exists(cfg.BLOCK_PATH):
            for f in os.listdir(cfg.BLOCK_PATH):
                os.remove(os.path.join(cfg.BLOCK_PATH, f))

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

class KMerger():
    """
    K-way merger for sorted posting lists stored on disk
    """

    def __init__(self, block_path: str):
        self.block_path = block_path

    def merge_blocks(self):
        cfg = Config(load=True)
        start = perf_counter()
        posting_path = os.path.join(cfg.DATA_PATH, "posting_list.bin")
        posting_list_file = open(posting_path, "w+b")
        posting_list_file.write(cfg.HEADER_POSTING_FILE.encode("utf-8"))

        index_path = os.path.join(cfg.DATA_PATH, "term_dictionary.bin")
        index_file = open(index_path, "w+b")
        index_file.write(cfg.HEADER_INDEX_FILE.encode("utf-8"))

        if os.path.exists(self.block_path):
            file_names = os.listdir(self.block_path)
            print(f"Merging {len(file_names)} blocks from {self.block_path}...")
            file_paths = [os.path.join(self.block_path, f) for f in file_names]

            terms_merged = 0

            for term, posting_list in self._merge_sorted_files(file_paths):
                disk_offset, length = self._add_posting_to_disk(posting_list_file, posting_list)
                self._add_index_entry_to_disk(index_file, term, disk_offset, length)
                terms_merged += 1
                if terms_merged % 100000 == 0:
                    print(f"Merged {terms_merged} terms...")
            print(f"Total merged terms: {terms_merged}")
            posting_list_file.close()
            index_file.close()
        else:
            print(f"No blocks found in {self.block_path} to merge.")
        end = perf_counter()
        print(f"Merge completed in {end - start:.2f} seconds.")

    def _add_posting_to_disk(self, posting_file ,posting_list: array) -> Tuple[int, int]:
        start = posting_file.tell()

        posting_file.write(struct.pack("<I", len(posting_list)))
        posting_file.write(posting_list.tobytes())

        end = posting_file.tell()
        return [start, end - start]

    def _add_index_entry_to_disk(self, index_file, term: str, disk_offset: int, posting_length: int) -> None:
        term = term.encode("utf-8")
        index_file.write(struct.pack("<I", len(term)))
        index_file.write(term)

        index_file.write(struct.pack("<Q", disk_offset))      
        index_file.write(struct.pack("<I", posting_length))  

    def _merge_sorted_files(self, paths: List[str]) -> Iterable[Tuple[str, array]]:
        files = [open(p, "rb") for p in paths]
        heap: List[Tuple[str, int, array.array[int]]] = []
        try:
            [self._read_magic_header(f) for f in files]

            for i, f in enumerate(files):
                term, arr = self._read_posting_line(f)
                if term:
                    heapq.heappush(heap, (term, i, arr))

            while heap:
                files_to_read = []

                term, i, arr = heapq.heappop(heap)
                files_to_read.append(i)
                # pop until different term
                while heap and heap[0][0] == term:
                    _, j, arr2 = heapq.heappop(heap)
                    arr.extend(arr2)
                    files_to_read.append(j)
                yield term, arr
                # push next from read files
                for i in files_to_read:
                    nxt_term, arr = self._read_posting_line(files[i])
                    if nxt_term:
                        heapq.heappush(heap, (nxt_term, i, arr))
        finally:
            for f in files:
                f.close()

    def _read_magic_header(self, file):
        magic_version = file.read(5)
        if magic_version != b"SEAB\x01":
            raise ValueError("Bad magic/version")
        


    def _read_posting_line(self, file):
        lb = file.read(4)
        if not lb:
            #TODO: delete file if reach EOF
            print("No more terms")
            return None, None
            # break
        (term_len,) = struct.unpack("<I", lb)
        term = file.read(term_len).decode("utf-8")
        (count,) = struct.unpack("<I", file.read(4))
        buf = file.read(count * 4)
        arr = array("I")
        arr.frombytes(buf)  # uint32 little-endian
        return term, arr

def main():
    cfg = Config(load=True)
    # ingestion = Ingestion(cfg.DOCUMENTS)
    # ingestion.ingest(cfg.INGESTION.NUM_DOCUMENTS, cfg.INGESTION.BATCH_SIZE)

    merger = KMerger(cfg.BLOCK_PATH)
    merger.merge_blocks()


if __name__ == "__main__":
    main()