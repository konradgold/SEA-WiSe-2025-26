import abc
from typing import List

import numpy
NUM_DOCS=3_300_000

class Ranking(abc.ABC):

    def __init__(self, cfg):
        self.num_docs = cfg.get('NUM_DOCS', NUM_DOCS)
        self.max_results = cfg.get('MAX_RESULTS', 10)
    
    def __call__(self, tokens):
        return self.rank(tokens)
    
    @abc.abstractmethod
    def _compute_score(self, token: dict) -> dict[int, float]:
        pass
    
    def rank(self, tokens: list[dict]) -> List[tuple[int, float]]:
        '''
        Docstring for rank
        
        :param self: Description
        :param results: Description
        :type results: dict
        :param tokens: Description
        :type tokens: List[dict]
        :return: Description
        :rtype: List[tuple[int, float]]

        Returns a score for each result based on the provided tokens.
        '''
        ranked_results = dict()
        for token in tokens:
            scores = self._compute_score(token)
            ranked_results.update({doc_id: ranked_results.get(doc_id, 0.0) + score 
                       for doc_id, score in scores.items()})
        return sorted(ranked_results.items(), key=lambda item: item[1], reverse=True)[:self.max_results]
        
class TFIDFRanking(Ranking):

    '''
    Docstring for TFIDFRanking
    Needs: {doc_id: pos_list or term_freq}
    '''
    def _compute_score(self, token: dict[int, List[int]] | dict[int, int]) -> dict[int, float]:
        result = dict()
        idf = numpy.log(self.num_docs/(len(token)+1))
        for doc_id, pos_list in token.items():
            if isinstance(pos_list, int):
                tf = numpy.log(1+pos_list)
            else:
                tf = numpy.log(1+len(pos_list))
            score = tf * idf
            result[doc_id] = score
        return result


class BM25Ranking(Ranking):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.avg_doc_len = cfg.get('AVG_DOC_LEN', 100.0)
        self.k1 = cfg.BM25.K1 if cfg.BM25.K1 is not None else 1.5
        self.b = cfg.BM25.B if cfg.BM25.B is not None else 0.75

    def _compute_score(self, token: dict[int, tuple[int, List[int]|int]]) -> dict[int, float]:
        '''
        :param token: {doc_id: (doc_len, pos_list or term_freq)}
        '''
        result = dict()
        idf = numpy.log((self.num_docs - len(token) + 0.5) / (len(token) + 0.5) + 1)
        for doc_id, pos_tuple in token.items():
            tf = len(pos_tuple[1]) if isinstance(pos_tuple[1], list) else pos_tuple[1]
            doc_len = pos_tuple[0]
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score = idf * ((tf * (self.k1 + 1)) / denom)
            result[doc_id] = score
        return result