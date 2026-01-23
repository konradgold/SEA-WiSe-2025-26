
from abc import abstractmethod
import json
from typing import List, Optional, Tuple
from sea.index.tokenization import TokenizerAbstract, get_tokenizer
import logging

from sea.query.splade import SpladeEncoder
from sea.ranking.io_wrapper import RankerAdapter
from sea.ranking.utils import Document

logger = logging.getLogger(__name__)

class AbstractOperator:
    
    @abstractmethod
    def execute(self, r, tokenizer) -> Tuple[Optional[List[Document]], str]:
        """
        r: Redis connection: Should not be openend/closed here, try avoid dos attack ;)
        tokenizer: Avoid constant re-initialization of tokenizer
        """
        pass
    
    @abstractmethod
    def tokenize(self, tokenizer) -> TokenizerAbstract:
        pass

class TermOperator(AbstractOperator):

    def __init__(self, phrase: str, splade_encoder: Optional[SpladeEncoder] = None):
        self.phrase = phrase
        self.splade_encoder = splade_encoder
            

    
    def tokenize(self, tokenizer: Optional[TokenizerAbstract]=None):
        if tokenizer is None:
            tokenizer = get_tokenizer()
        self.tokens = tokenizer.tokenize(self.phrase)
        return tokenizer

    def execute(self, r: RankerAdapter, tokenizer: Optional[TokenizerAbstract]=None) -> Tuple[Optional[List[Document]], str]:
        if not hasattr(self, 'tokens'):
            self.tokenize(tokenizer)
        if len(self.tokens) == 0:
            logging.warning("No tokens found. Phrase might have been eaten by tokenizer.")
            return None, ""
        if self.splade_encoder is not None:
            self.tokens = self.splade_encoder.expand(" ".join(self.tokens))
        result = r(self.tokens)
        logging.debug(f"Term Operator result: {result}")
        return result, " ".join(self.tokens)
    

class PhraseOperator(AbstractOperator):

    _seq_len: int  # Required matched sequence length, defaults to 2

    def __init__(self, phrase: str, seq_len: Optional[int] = None):
        self.seq_len = seq_len
        self.phrase = phrase

    @property
    def seq_len(self):
        if self._seq_len is None:
            self._seq_len = 2
        return self._seq_len

    @seq_len.setter
    def seq_len(self, value: Optional[int]):
        if value is None or value < 1:
            self._seq_len = 2
            return
        if hasattr(self, 'tokens') and value > len(self.tokens):
            self._seq_len = len(self.tokens)
        self._seq_len = value

    def tokenize(self, tokenizer: Optional[TokenizerAbstract]=None):
        if tokenizer is None:
            tokenizer = get_tokenizer()
        self.tokens = tokenizer.tokenize(self.phrase)
        if len(self.tokens) < self.seq_len:
            self.seq_len = len(self.tokens)
        return tokenizer

    def execute(self, r, tokenizer: Optional[TokenizerAbstract]=None) -> Tuple[Optional[List[Document]], str]:
        raise NotImplementedError("Use execute_phrase_matching for phrase matching.")
        if not hasattr(self, 'tokens'):
            self.tokenize(tokenizer)
        if len(self.tokens) == 0:
            logging.warning("No tokens found. Phrase might have been eaten by tokeinzer.")
            return None
        assert self.seq_len <= len(self.tokens), "Sequence length cannot be greater than number of tokens"
        tokens = [f"token:{t}" for t in self.tokens if t]
        results = self._get_positions(r, tokens)

        if len(results) < self.seq_len:
            return set()  # Not enough tokens found
        all_matching_docs = set()
        for i in range(len(results) - self.seq_len + 1):
            window_matching_docs = set()
            window_positions = results[i:i + self._seq_len]
            common_docs = set(window_positions[0].keys())

            for result_dict in window_positions[1:]:
                common_docs.intersection_update(result_dict.keys())
            if not common_docs:
                return set() # phrase not found, early exit

            for doc_id in common_docs:
                if self._sequence_exists(window_positions, doc_id):
                    window_matching_docs.add(doc_id)
            if not all_matching_docs:
                all_matching_docs = window_matching_docs
            else:
                all_matching_docs.intersection_update(window_matching_docs)
        return all_matching_docs
    
    def _sequence_exists(self, window_positions, doc_id) -> bool:
        positions_lists = [result_dict[doc_id] for result_dict in window_positions] # contains list of positions for each token in the window for the respective doc_id, sorted
        for pos in positions_lists[0]:
            is_consecutive = True
            for j in range(1, self.seq_len):
                if not (pos + j) in positions_lists[j]:
                    is_consecutive = False
                    break
            if is_consecutive:
                return True
        return False

    def _get_positions(self, r, tokens: List[str]) -> List[dict]:
        results = []
        for token in tokens:
            inv_result = r.hgetall(token)
            if inv_result:
                try:
                    inv_result = {key: json.loads(value)['pos'] for key, value in inv_result.items()} # Assumes positions are stored and under 'pos'
                    results.append(inv_result)
                except (AttributeError, KeyError):
                    logging.warning(f"Positions not found for token: {token}")
                    return []
            else:
                return []  # early exit if any token not found
        return results

class OROperator(AbstractOperator):

    def __init__(self, children: List[AbstractOperator]):
        self.children = children

    def execute(self, r, tokenizer: Optional[TokenizerAbstract] = None) -> Tuple[Optional[List[Document]], str]:
        result = {}
        final_query = ""
        if tokenizer is None:
            tokenizer = get_tokenizer()
        for child in self.children:
            if isinstance(child, PhraseOperator):
                continue
            results, query = child.execute(r, tokenizer)
            final_query += " or " + query if final_query else query
            if results is not None:
                for doc in results:
                    if doc.doc_id in result:
                        result[doc.doc_id]["score"] += doc.score
                    else:
                        result[doc.doc_id] = doc
        print(f"OR Operator result: {result}")
        return list(result.values()), final_query

class ANDOperator(AbstractOperator):

    def __init__(self, children: List[AbstractOperator], splade_encoder=None):
        self.children = children
        self.splade_encoder = splade_encoder

    def execute(self, r, tokenizer: Optional[TokenizerAbstract]=None) -> Tuple[Optional[List[Document]], str]:
        intersection = {}
        result_1 = {}
        final_query = ""
        if tokenizer is None:
            tokenizer = get_tokenizer()
        for child in self.children:
            if len(intersection) == 0:
                result, query = child.execute(r, tokenizer)
                if result is not None:
                    intersection = {doc.doc_id: doc for doc in result}
                else:
                    return None, ""
                final_query += " and " + query if final_query else query
            else:
                result_2, query = child.execute(r, tokenizer)
                if result_2 is not None:
                    result_2 = {doc.doc_id: doc for doc in result_2}
                    intersection = {doc_id: intersection[doc_id] for doc_id in intersection if doc_id in result_2}
                    for doc_id in intersection:
                        intersection[doc_id].score += result_2[doc_id].score
                final_query += " and " + query if final_query else query
            
        print(f"AND Operator result: {intersection}")
        return list(intersection.values()), final_query
    

class ANDNOTOperator(AbstractOperator):

    def __init__(self, children: List[AbstractOperator]):
        self.children = children
        assert len(children) == 2, "AndNotOperator requires exactly two children"

    def execute(self, r, tokenizer: Optional[TokenizerAbstract]=None) -> Tuple[Optional[List[Document]], str]:
        if tokenizer is None:
            tokenizer = get_tokenizer()
        try:
            result_1, query1 = self.children[0].execute(r, tokenizer)
            result_1 = {doc.doc_id: doc for doc in result_1}
            result_2, query2 = self.children[1].execute(r, tokenizer)
            result_2 = {doc.doc_id: doc for doc in result_2}
            result = {id: doc for id, doc in result_1.items() if id not in result_2}
            return list(result.values()), f"{query1} and not {query2}"
        except Exception:
            return [], ""