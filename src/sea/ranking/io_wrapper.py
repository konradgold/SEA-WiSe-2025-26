import abc
import os
import pickle
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import List, Optional

import tqdm
from omegaconf import DictConfig

from sea.ranking.ranking import BM25Ranking, Ranking, TFIDFRanking
from sea.ranking.utils import Document, RankingRegistry
from sea.storage.manager import StorageManager
from sea.utils.config_wrapper import Config

def build_index(tsv_path: str, interval: int = 1000, index_path: str = 'offsets.pkl', limit: int = -1) -> list[int]:
    """Build an offset index for fast random access to TSV rows."""
    offsets = [0]
    with open(tsv_path, 'rb') as f:
        for i, _ in tqdm.tqdm(enumerate(f, 1)):
            if i % interval == 0:
                offsets.append(f.tell())
            if limit > 0 and i >= limit:
                break

    with open(index_path, 'wb') as f:
        pickle.dump(offsets, f)
    return offsets


def get_default_thread_count() -> int:
    """Calculate default thread count for parallel document reading."""
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


class RankerAdapter(abc.ABC):

    def __init__(self, ranker: Ranking, cfg: Optional[DictConfig] = None, num_threads: Optional[int] = None):
        if cfg is None:
            cfg = Config(load=True)
        self.ranker = ranker
        self.cfg = cfg
        self.max_results = cfg.SEARCH.MAX_RESULTS or 10
        self.cut = cfg.SEARCH.POSTINGS_CUT or 100
        self.fields = cfg.SEARCH.FIELDED.FIELDS if cfg.SEARCH.FIELDED.ACTIVE else ["all"]
        self.interval = cfg.INDEX_INTERVAL

        self._init_storage_managers(cfg)
        self._init_offsets(cfg)

        self.num_threads = num_threads if num_threads is not None else get_default_thread_count()
        if self.num_threads > 1 and cfg.SEARCH.VERBOSE_OUTPUT:
            print(f"Using {self.num_threads} threads for document reading.")

    def _init_storage_managers(self, cfg: DictConfig) -> None:
        """Initialize storage managers for each field."""
        self.storage_managers = {}
        for field in self.fields:
            storage_field = None if field == "all" else field
            storage_manager = StorageManager(rewrite=False, cfg=cfg, field=storage_field)
            storage_manager.init_all()
            self.storage_managers[field] = storage_manager

    def _init_offsets(self, cfg: DictConfig) -> None:
        """Load or build document offsets index.

        Automatically rebuilds if the existing offsets file doesn't cover
        all documents in the index (e.g., if NUM_DOCUMENTS changed).
        """
        needs_rebuild = False

        if os.path.exists(cfg.DOCUMENT_OFFSETS):
            with open(cfg.DOCUMENT_OFFSETS, 'rb') as f:
                self.offsets = pickle.load(f)

            # Check if offsets cover all docs in the index
            max_doc_id = max(self.storage_managers[self.fields[0]].getDocMetadata().keys())
            max_row_covered = (len(self.offsets) - 1) * cfg.INDEX_INTERVAL + cfg.INDEX_INTERVAL - 1
            if max_doc_id > max_row_covered:
                print(f"Offsets file stale (covers {max_row_covered} rows, need {max_doc_id}). Rebuilding...")
                needs_rebuild = True
        else:
            needs_rebuild = True

        if needs_rebuild:
            print("Building document offsets index...")
            self.offsets = build_index(
                tsv_path=cfg.DOCUMENTS,
                interval=cfg.INDEX_INTERVAL,
                index_path=cfg.DOCUMENT_OFFSETS,
                limit=cfg.INGESTION.NUM_DOCUMENTS
            )

    def __call__(self, tokens: list) -> list[Document]:
        return self._retrieve_and_rank(tokens)

    def _retrieve_and_rank(self, tokens: List[str]) -> list[Document]:
        ranked_results: dict[int, float] = {}
        for field in self.fields:
            token_list = self._prepare_tokens(tokens, field=field)
            if not token_list:
                continue
            for doc_id, score in self.ranker(token_list, field=field).items():
                ranked_results[doc_id] = ranked_results.get(doc_id, 0.0) + score

        if not ranked_results:
            return []

        results = sorted(ranked_results.items(), key=lambda x: x[1], reverse=True)[:self.max_results]
        return self._read_documents(results)

    def _prepare_tokens(self, tokens: List[str], field: str) -> list:
        token_list = []
        for token in tokens:
            posting_list: array | None = self.storage_managers[field].getPostingList(token)
            if posting_list is None:
                continue
            processed_token = self.process_posting_list(posting_list, field=field)
            if processed_token:
                token_list.append(processed_token)
        return token_list

    def _get_lines(self, row_numbers: list[int]) -> List[List[str]]:
        """Read document lines using parallel threads."""
        t0 = perf_counter()
        sorted_rows = sorted(row_numbers)

        # Divide rows among threads
        batch_size = max(1, len(sorted_rows) // self.num_threads)
        batches = [
            sorted_rows[i:i + batch_size]
            for i in range(0, len(sorted_rows), batch_size)
        ]

        all_results = {}
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._fetch_batch, batch) for batch in batches]
            for future in as_completed(futures):
                all_results.update(future.result())

        documents_output = [all_results[row] for row in row_numbers]

        if self.cfg.SEARCH.VERBOSE_OUTPUT:
            elapsed = (perf_counter() - t0) * 1000
            print(f"Reading {len(row_numbers)} lines took {elapsed:.2f} ms ({elapsed/len(row_numbers):.2f} ms/line)")

        return documents_output

    def _fetch_batch(self, batch_rows: list[int]) -> dict[int, list[str]]:
        """Fetch a batch of document rows from disk."""
        results = {}
        max_row = (len(self.offsets) - 1) * self.interval + self.interval - 1
        with open(self.cfg.DOCUMENTS, 'rb') as f:
            for row_num in batch_rows:
                index = row_num // self.interval
                steps = row_num % self.interval

                if index >= len(self.offsets):
                    raise IndexError(
                        f"Document row {row_num} exceeds offsets coverage (max ~{max_row}). "
                        f"Delete {self.cfg.DOCUMENT_OFFSETS} to rebuild."
                    )

                f.seek(self.offsets[index] - f.tell(), 1)
                for _ in range(steps):
                    f.readline()

                line = f.readline()
                results[row_num] = line.decode('utf-8').rstrip('\n').split('\t')
        return results

    def _read_documents(self, ranked_results: list[tuple[int, float]]) -> list[Document]:
        if self.cfg.SEARCH.VERBOSE_OUTPUT:
            print(f"Reading {len(ranked_results)} documents from disk...")

        id_to_score = dict(ranked_results)
        row_numbers = [doc_id for doc_id, _ in ranked_results]
        documents_output = self._get_lines(row_numbers)

        return [
            Document(
                doc_id=row[0],
                link=row[1],
                title=row[2],
                content=row[3],
                score=id_to_score[row_num],
            )
            for row_num, row in zip(row_numbers, documents_output)
        ]

    @abc.abstractmethod
    def process_posting_list(self, pl: array, field: Optional[str] = None) -> dict:
        pass


class TFIDF(RankerAdapter):

    def __init__(self, ranker: Ranking, cfg: Optional[DictConfig] = None, num_threads: Optional[int] = None):
        super().__init__(ranker, cfg, num_threads)

    def process_posting_list(self, pl: array, field: Optional[str] = None) -> dict[int, list[int]]:
        pos_list = pl.tolist()
        if not pos_list:
            return {}
        len_pl = pos_list.pop(0)
        assert len_pl == len(pos_list)
        posting_dict = {}
        i = 0
        while i+1 < len_pl:
            posting_dict[pos_list[i]] = pos_list[i+1]
            i += 2 # Assuming that positions are not stored
        return posting_dict

class BM25(RankerAdapter):

    def __init__(self, ranker: Ranking, cfg: Optional[DictConfig] = None, num_threads: Optional[int] = None):
        super().__init__(ranker, cfg, num_threads)

    def process_posting_list(self, pl: array, field: Optional[str] = None) -> dict[int, tuple[int, int]]:
        pos_list = pl.tolist()
        if not pos_list:
            return {}
        len_pl = pos_list.pop(0)
        assert len_pl == len(pos_list)
        posting_dict = {}
        i = 0
        doc_ids = []
        tfs = []
        while i+1 < len_pl:
            doc_ids.append(pos_list[i])
            tfs.append(pos_list[i+1])
            i += 2 # Assuming that positions are not stored
        if len(doc_ids) > self.cut:
            if self.cfg.SEARCH.VERBOSE_OUTPUT:
                print(
                    f"Skipping token with {len(doc_ids)} postings exceeding cut of {self.cut}."
                )
            return {}
        for doc_id, tf in zip(doc_ids, tfs):
            doc_len = self.storage_managers[field].getDocMetadataEntry(doc_id)[1]
            posting_dict[doc_id] = (doc_len, tf)
        return posting_dict

RankersRegistry = RankingRegistry()


def bm25(cfg: Optional[DictConfig] = None, num_threads: Optional[int] = None) -> BM25:
    if cfg is None:
        cfg = Config(load=True)
    ranker = BM25Ranking(cfg)
    return BM25(ranker, cfg=cfg, num_threads=num_threads)


def tfidf(cfg: Optional[DictConfig] = None, num_threads: Optional[int] = None) -> TFIDF:
    if cfg is None:
        cfg = Config(load=True)
    ranker = TFIDFRanking(cfg)
    return TFIDF(ranker, cfg=cfg, num_threads=num_threads)


RankersRegistry.register("bm25", bm25)
RankersRegistry.register("tfidf", tfidf)
