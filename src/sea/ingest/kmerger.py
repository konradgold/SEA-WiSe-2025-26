from array import array
import heapq
import os
from time import perf_counter
from typing import Iterable, List, Tuple
from sea.storage.IO import BlockIO
from sea.storage.manager import StorageManager


class KMerger():
    """
    K-way merger for sorted posting lists stored on disk
    """

    def __init__(self, block_path: str, fields: List[str] = ["all"]):
        
        self.block_paths: dict[str, str] = {}
        self.blockIOs: dict[str, BlockIO] = {}
        self.storageManagers: dict[str, StorageManager] = {}
        for field in fields:
            self.block_paths[field] = block_path + (f"{field}/" if field is not None else "")
            self.blockIOs[field] = BlockIO(field=field)
            self.storageManagers[field] = StorageManager(rewrite=True, rewrite_doc_dict=True, field=field)
    
    def merge_blocks(self):
        start = perf_counter()
        terms_merged = 0
        for field in self.block_paths:
            self.block_path = self.block_paths[field]
            self.storageManager = self.storageManagers[field]
            print(f"Merging blocks for field '{field}' from path '{self.block_path}'...")
            if os.path.exists(self.block_path):
                file_names = os.listdir(self.block_path)
                print(f"Merging {len(file_names)} blocks from {self.block_path}...")
                file_paths = [os.path.join(self.block_path, f) for f in file_names]

                for term, posting_list in self._merge_sorted_files(file_paths, field):
                    self.storageManager.write_term_posting_list(term, posting_list)
                    terms_merged += 1
                    if terms_merged % 100000 == 0:
                        print(f"Merged {terms_merged:,} terms...")
                self.storageManager.close()
            else:
                print(f"No blocks found in {self.block_path} to merge.")
        end = perf_counter()
        total_time = end - start
        return terms_merged, total_time

    def _merge_sorted_files(self, paths: List[str], field: str) -> Iterable[Tuple[str, array]]:
        files = [open(p, "rb") for p in paths]
        heap: List[Tuple[str, int, array[int]]] = []
        try:
            [self.blockIOs[field].check_magic_header(f) for f in files]

            for i, f in enumerate(files):
                term, arr = self.blockIOs[field].read_line(f)
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
                    nxt_term, arr = self.blockIOs[field].read_line(files[i])
                    if nxt_term:
                        heapq.heappush(heap, (nxt_term, i, arr))
        finally:
            for f in files:
                f.close()
