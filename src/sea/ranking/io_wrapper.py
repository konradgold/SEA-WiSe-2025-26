import abc
from array import array
import enum
from typing import List, Optional

from regex import R
from sea.utils.config import Config
from sea.ranking.utils import Document, RankingRegistry
from sea.ranking.ranking import BM25Ranking, Ranking, TFIDFRanking
from sea.storage.manager import StorageManager
import pandas as pd

def read_tsv_rows(tsv_path: str, row_numbers: list[int], sep='\t', header=0) -> pd.DataFrame:
    """Load specific rows by 0-indexed line numbers, handles header correctly."""
    adjusted_rows = [r + (1 if header else 0) for r in row_numbers]  # +1 for header line
    
    # Skip all rows except targets
    skiprows = lambda x: x not in adjusted_rows
    return pd.read_csv(tsv_path, sep=sep, skiprows=skiprows, header=header)


class RankerAdapter(abc.ABC):

    def __init__(self, ranker: Ranking, storage_manager: StorageManager, cfg: Optional[Config] = None):
        self.ranker = ranker
        if cfg is None:
            cfg = Config(load=True)
        self.cfg = cfg
        self.storage_manager = storage_manager
        self.storage_manager.init_all()

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
            token_list.append(self.process_posting_list(posting_list))
        return token_list
    
    def _read_documents(self, ranked_results: list[tuple[int, float]]) -> list[Document]:
        ranked_results.sort(key=lambda x: x[0])
        row_numbers = [doc_line for doc_line, _ in ranked_results]
        df = read_tsv_rows(
            self.cfg.DOCUMENTS, 
            row_numbers)  # Sorted by file
        
        documents_output = []
        for i, (_, row) in enumerate(df.iterrows()):
            parts =row.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            documents_output.append(
                Document(
                    doc_id=parts[0],
                    link=parts[1],
                    title=parts[2],
                    content=parts[3],
                    score=ranked_results[i][1]
                )
            )
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
        while i < len_pl+1:
            doc_id = pos_list[i]
            tf = pos_list[i+1]
            doc_len = self.storage_manager.getDocMetadataEntry(doc_id)[1]
            posting_dict[doc_id] = (doc_len, tf)
            i += pos_list[i+1] + 2
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