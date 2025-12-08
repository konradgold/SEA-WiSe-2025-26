import abc
from array import array
from time import perf_counter
from typing import List, Optional
import polars as pl
from sea.utils.config import Config
from sea.ranking.utils import Document, RankingRegistry
from sea.ranking.ranking import BM25Ranking, Ranking, TFIDFRanking
from sea.storage.manager import StorageManager
import pandas as pd
from io import StringIO

def read_tsv_rows(tsv_path: str, row_numbers: list[int], sep='\t', header=None) -> pd.DataFrame:
    row_numbers = sorted(row_numbers)
    
    target_lines = []
    header_line = None

    
    with open(tsv_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Handle header
        if header is not None:
            header_line = next(f)
        
        # Jump directly to target lines
        target_idx = 0
        for i, line in enumerate(f):
            if target_idx < len(row_numbers) and i == row_numbers[target_idx]:
                target_lines.append(line)
                target_idx += 1
            if target_idx >= len(row_numbers):
                break  # Found all targets, STOP READING!
    
    # Build content
    content = (header_line or '') + ''.join(target_lines)
    return pd.read_csv(StringIO(content), sep=sep, header=header)

class RankerAdapter(abc.ABC):

    def __init__(self, ranker: Ranking, storage_manager: StorageManager, cfg: Optional[Config] = None):
        self.ranker = ranker
        if cfg is None:
            cfg = Config(load=True)
        self.cfg = cfg
        self.storage_manager = storage_manager
        self.storage_manager.init_all()
        print("Loading document offsets into memory...")
        df = pl.scan_csv(self.cfg.DOCUMENTS, separator='\t', infer_schema_length=0)
        print("Collecting document offsets...")
        self.row_offsets = df.select(pl.int_range(0, 100000).alias('row')).collect()
        

    def __call__(self, tokens: list) -> list[Document]:
        return self._retrieve_and_rank(tokens)

    def _retrieve_and_rank(self, tokens: List[str]) -> list[Document]:
        token_list = self._prepare_tokens(tokens)
        if len(token_list) == 0:
            return []
        ranked_results = self.ranker(token_list)
        return self._read_documents(ranked_results)

    def _prepare_tokens(self, tokens: List[str]) -> list:
        token_list = []
        for token in tokens:
            posting_list: array | None = self.storage_manager.getPostingList(token)
            if posting_list is None:
                continue
            processed_token = self.process_posting_list(posting_list)
            if processed_token:
                token_list.append(processed_token)
        return token_list
    
    def _read_documents(self, ranked_results: list[tuple[int, float]]) -> list[Document]:
        print(f"Reading {len(ranked_results)} documents from disk...")
        ranked_results.sort(key=lambda x: x[0])
        row_numbers = [doc_line for doc_line, _ in ranked_results]
        t0 = perf_counter()
        target_rows = self.row_offsets.filter(pl.col('row').is_in(row_numbers))
        target_rows = target_rows.sort('row').collect()
        t1 = perf_counter()
        print(f"Read {len(ranked_results)} documents in {(t1 - t0)*1000:.1f} ms")
        
        documents_output = []
        # Assuming df columns: doc_id, link, title, content
        valid_mask = target_rows.notna().sum(axis=1) == 4
        valid_docs = target_rows[valid_mask].head(len(ranked_results)).reset_index(drop=True)

        documents_output = [
            Document(
                doc_id=row[0], link=row[1], title=row[2], content=row[3],
                score=ranked_results[i][1]
            )
            for i, row in enumerate(valid_docs.itertuples(index=False))
        ]

        doc_outputs = sorted(documents_output, reverse=True)
        return doc_outputs
    
    @abc.abstractmethod
    def process_posting_list(self, pl: array) -> dict:
        pass


class TFIDF(RankerAdapter):
    
    def process_posting_list(self, pl: array) -> dict[int, list[int]]:
        pos_list = pl.tolist()
        len_pl = pos_list.pop(0)
        assert len_pl == len(pos_list)
        posting_dict = {}
        i = 0
        while i < len_pl+1:
            posting_dict[pos_list[i]] = pos_list[i+1]
            i += pos_list[i+1] + 2
        return posting_dict

    
class BM25(RankerAdapter):     
    
    def process_posting_list(self, pl: array) -> dict[int, tuple[int, int]]:
        pos_list = pl.tolist()
        len_pl = pos_list.pop(0)
        assert len_pl == len(pos_list)
        posting_dict = {}
        i = 0
        while i+1 < len_pl:
            doc_id = pos_list[i]
            tf = pos_list[i+1]
            doc_len = self.storage_manager.getDocMetadataEntry(doc_id)[1]
            posting_dict[doc_id] = (doc_len, tf)
            i += pos_list[i+1] + 2
        if len(posting_dict) > 1000:
            return {}
            print(f"Warning: Large posting list with {len(posting_dict)} entries encountered.")
        return posting_dict
    

RankersRegistry = RankingRegistry()

def bm25():
    cfg = Config(load=True)
    storage_manager = StorageManager(rewrite=False, cfg=cfg)
    ranker = BM25Ranking(cfg)
    return BM25(ranker, storage_manager, cfg=cfg)

def tfidf():
    cfg = Config(load=True)
    storage_manager = StorageManager(rewrite=False, cfg=cfg)
    ranker = TFIDFRanking(cfg)
    return TFIDF(ranker, storage_manager, cfg=cfg)


RankersRegistry.register("bm25", bm25)
RankersRegistry.register("tfidf", tfidf)