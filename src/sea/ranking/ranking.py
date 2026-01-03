import abc
from typing import List, Optional

import numpy
NUM_DOCS=3_300_000

class Ranking(abc.ABC):

    def __init__(self, cfg):
        self.num_docs = cfg.SEARCH.NUM_DOCS if cfg.SEARCH.NUM_DOCS is not None else NUM_DOCS
        self.weights = cfg.SEARCH.FIELDED.WEIGHTS if cfg.SEARCH.FIELDED.ACTIVE else {"all": 1.0}
    
    def __call__(self, tokens, field: str) -> dict[int, float]:
        return self.rank(tokens, field)
    
    @abc.abstractmethod
    def _compute_score(self, token: dict, field: str) -> dict[int, float]:
        pass
    
    def rank(self, tokens: list[dict], field: str) -> dict[int, float]:
        '''
        Ranks documents based on the provided tokens.  

        :param tokens: A list of token dictionaries, each representing query terms and their associated data for scoring.  
        :type tokens: list[dict]  
        :return: A list of tuples (doc_id, score), sorted by score in descending order, limited to the maximum number of results.  
        :rtype: List[tuple[int, float]] 
        '''
        ranked_results = dict()
        for token in tokens:
            scores = self._compute_score(token, field=field)
            ranked_results.update({doc_id: ranked_results.get(doc_id, 0.0) + score 
                       for doc_id, score in scores.items()})
        return ranked_results
        
class TFIDFRanking(Ranking):

    '''
    Docstring for TFIDFRanking
    Needs: {doc_id: pos_list or term_freq}
    '''
    def _compute_score(self, token: dict[int, List[int]] | dict[int, int], field: str) -> dict[int, float]:
        result = dict()
        idf = numpy.log(self.num_docs/(len(token)+1))
        for doc_id, pos_list in token.items():
            if isinstance(pos_list, int):
                tf = numpy.log(1+pos_list) 
            else:
                tf = numpy.log(1+len(pos_list))
            score = tf * idf * self.weights.get(field, 1.0)
            result[doc_id] = score
        return result


class BM25Ranking(Ranking):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.avg_doc_len = cfg.SEARCH.FIELDED.LENGTHS if cfg.SEARCH.FIELDED.ACTIVE else {"all": cfg.SEARCH.AVG_DOC_LEN}
        self.k1 = cfg.BM25.K1 if cfg.BM25.K1 is not None else 1.5
        self.b = cfg.BM25.B if cfg.BM25.B is not None else 0.75

    def _compute_score(self, token: dict[int, tuple[int, List[int]|int]], field: str) -> dict[int, float]:
        '''
        :param token: {doc_id: (doc_len, pos_list or term_freq)}
        '''
        result = dict()
        idf = numpy.log((self.num_docs - len(token) + 0.5) / (len(token) + 0.5) + 1)
        for doc_id, pos_tuple in token.items():
            tf = len(pos_tuple[1]) if isinstance(pos_tuple[1], list) else pos_tuple[1]
            tf *= self.weights.get(field, 1.0)
            doc_len = pos_tuple[0]
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len.get(field, self.avg_doc_len.get("all", 100.0))))
            score = idf * ((tf * (self.k1 + 1)) / denom)
            result[doc_id] = score
        return result