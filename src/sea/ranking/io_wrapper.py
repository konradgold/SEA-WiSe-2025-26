import abc
from array import array
from time import perf_counter
from typing import List, Optional
import pickle
import tqdm
from sea.utils.config_wrapper import Config
from omegaconf import DictConfig
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sea.ranking.utils import Document, RankingRegistry
from sea.ranking.ranking import BM25Ranking, Ranking, TFIDFRanking
from sea.storage.manager import StorageManager

def build_index(tsv_path, interval=1000, index_path='offsets.pkl', limit=-1):
    offsets = []
    with open(tsv_path, 'rb') as index:
        offsets.append(0)
        for i, line in tqdm.tqdm(enumerate(index, 1)):
            if i % interval == 0:
                offsets.append(index.tell())
            if limit > 0 and i >= limit:
                break
    
    with open(index_path, 'wb') as index:
        pickle.dump(offsets, index)
    return offsets


class RankerAdapter(abc.ABC):

    def __init__(self, ranker: Ranking, cfg: Optional[DictConfig] = None):
        if cfg is None:
            cfg = Config(load=True)
        self.ranker = ranker
        self.max_results = cfg.SEARCH.MAX_RESULTS if cfg.SEARCH.MAX_RESULTS is not None else 10
        self.cut = (
            cfg.SEARCH.POSTINGS_CUT if cfg.SEARCH.POSTINGS_CUT is not None else 100
        )
        self.cfg = cfg
        if cfg.SEARCH.FIELDED.ACTIVE:
            self.fields = cfg.SEARCH.FIELDED.FIELDS
        else:
            self.fields = ["all"]
        self.storage_managers = {}
        for field in self.fields:
            if field == "all":
                field = None
            storage_manager = StorageManager(rewrite=False, cfg=cfg, field=field)
            storage_manager.init_all()
            if field is None:
                field = "all"
            self.storage_managers[field] = storage_manager
        if os.path.exists(cfg.DOCUMENT_OFFSETS):
            with open(cfg.DOCUMENT_OFFSETS, 'rb') as index:
                self.offsets = pickle.load(index)
        else:
            print("Building document offsets index...")
            self.offsets = build_index(
                tsv_path=cfg.DOCUMENTS,
                interval=cfg.INDEX_INTERVAL,
                index_path=cfg.DOCUMENT_OFFSETS,
                limit=cfg.INGESTION.NUM_DOCUMENTS
            )
                                        
        self.interval = cfg.INDEX_INTERVAL
        cpu_count = os.cpu_count()
        if cpu_count is None or cpu_count <=2:
            self.num_threads = 1
        else:
            self.num_threads = cpu_count - 2
        print(f"Using {self.num_threads} threads for document reading.")

    def __call__(self, tokens: list) -> list[Document]:
        return self._retrieve_and_rank(tokens)

    def _retrieve_and_rank(self, tokens: List[str]) -> list[Document]:
        ranked_results: dict[int, float] = dict()
        for field in self.fields:
            token_list = self._prepare_tokens(tokens, field=field)
            if len(token_list) == 0:
                continue
            for doc_id, score in self.ranker(token_list, field=field).items():
                ranked_results[doc_id] = ranked_results.get(doc_id, 0.0) + score
        if len(ranked_results) == 0:
            return []
        results = sorted(ranked_results.items(), key=lambda item: item[1], reverse=True)[:self.max_results]
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
        """Multithreaded batched version - divides sorted rows among threads"""

        def fetch_batch(batch_rows):
            """Each thread processes a batch sequentially with one file handle"""
            results = {}
            with open(self.cfg.DOCUMENTS, 'rb') as f:

                for row_num in batch_rows:
                    index = row_num // self.interval
                    steps = row_num % self.interval

                    f.seek(self.offsets[index]-f.tell(), 1)

                    for _ in range(steps):
                        f.readline()

                    line = f.readline()
                    results[row_num] = line.decode('utf-8').rstrip('\n').split('\t')

            return results

        t0 = perf_counter()

        # Sort once
        sorted_rows = sorted(row_numbers)

        # Divide into batches for each thread

        batch_size = len(sorted_rows) // self.num_threads
        batches = []

        for i in range(self.num_threads):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < self.num_threads - 1 else len(sorted_rows)
            if start_idx < len(sorted_rows):
                batches.append(sorted_rows[start_idx:end_idx])
        # Process batches in parallel
        all_results = {}
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(fetch_batch, batch) for batch in batches]

            for future in as_completed(futures):
                batch_results = future.result()
                all_results.update(batch_results)

        # Return in original order
        documents_output = [all_results[i] for i in row_numbers]

        if self.cfg.SEARCH.VERBOSE_OUTPUT:
            elapsed = (perf_counter() - t0) * 1000
            print(
                f"Reading {len(row_numbers)} lines took {elapsed:.2f} ms ({elapsed/len(row_numbers):.2f} ms/line)"
            )

        return documents_output

    def _read_documents(self, ranked_results: list[tuple[int, float]]) -> list[Document]:
        if self.cfg.SEARCH.VERBOSE_OUTPUT:
            print(f"Reading {len(ranked_results)} documents from disk...")

        # Save original order and map internal ID to score
        id_to_score = {doc_id: score for doc_id, score in ranked_results}
        row_numbers = [doc_id for doc_id, _ in ranked_results]

        # Fetch documents from disk (this might return them in different order depending on implementation,
        # but RankerAdapter._get_lines is supposed to return them in the order of row_numbers)
        documents_output = self._get_lines(row_numbers)

        hydrated_docs = [
            Document(
                doc_id=row[0],
                link=row[1],
                title=row[2],
                content=row[3],
                score=id_to_score[row_num],
            )
            for row_num, row in zip(row_numbers, documents_output)
        ]

        return hydrated_docs

    @abc.abstractmethod
    def process_posting_list(self, pl: array, field: Optional[str] = None) -> dict:
        pass


class TFIDF(RankerAdapter):

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


def bm25(cfg: Optional[DictConfig] = None):
    if cfg is None:
        cfg = Config(load=True)
    ranker = BM25Ranking(cfg)
    return BM25(ranker, cfg=cfg)


def tfidf(cfg: Optional[DictConfig] = None):
    if cfg is None:
        cfg = Config(load=True)
    ranker = TFIDFRanking(cfg)
    return TFIDF(ranker, cfg=cfg)


RankersRegistry.register("bm25", bm25)
RankersRegistry.register("tfidf", tfidf)
