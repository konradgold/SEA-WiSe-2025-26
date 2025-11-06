
from abc import abstractmethod
import json
from typing import List, Optional, Sequence

from zmq import has
from sea.index.tokenization import TokenizerAbstract, get_tokenizer
import logging

logger = logging.getLogger(__name__)

class AbstractOperator:
    
    @abstractmethod
    def execute(self, r, tokenizer) -> Optional[set]:
        """
        r: Redis connection: Should not be openend/closed here, try avoid dos attack ;)
        tokenizer: Avoid constant re-initialization of tokenizer
        """
        pass
    
    @abstractmethod
    def tokenize(self, tokenizer) -> TokenizerAbstract:
        pass

class TermOperator(AbstractOperator):

    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def tokenize(self, tokenizer: Optional[TokenizerAbstract]=None):
        if tokenizer is None:
            tokenizer = get_tokenizer()
        self.tokens = tokenizer.tokenize(self.phrase)
        return tokenizer

    def execute(self, r, tokenizer: Optional[TokenizerAbstract]=None) -> Optional[set]:
        if not hasattr(self, 'tokens'):
            self.tokenize(tokenizer)
        if len(self.tokens) == 0:
            logging.warning("No tokens found. Phrase might have been eaten by tokeinzer.")
            return None
        tokens = [f"token:{t}" for t in self.tokens if t]
        result = set()
        for token in tokens:
            inv_result = r.hgetall(token)

            if inv_result:
                inv_result = set(key for key in inv_result.keys())
                if not result:
                    result = inv_result
                else:
                    result = result.intersection(inv_result)
            else:
                return set()  # early exit if any token not found
        logging.debug(f"Term Operator result: {result}")
        return result

class PhraseOperator(AbstractOperator):

    _seq_len: int  # Required matched sequence length, defaults to 2

    def __init__(self, phrase: str, seq_len: Optional[int] = None):
        self._seq_len = seq_len if seq_len is not None else 2
        self.phrase = phrase

    @property
    def seq_len(self):
        if self._seq_len is None:
            self._seq_len = 2
        return self._seq_len

    @seq_len.setter
    def seq_len(self, value: int):
        if value is None or value < 1:
            self._seq_len = 2
        if hasattr(self, 'tokens') and value > len(self.tokens):
            self._seq_len = len(self.tokens)
        self._seq_len = value

    def tokenize(self, tokenizer: Optional[TokenizerAbstract]=None):
        if tokenizer is None:
            tokenizer = get_tokenizer()
        self.tokens = tokenizer.tokenize(self.phrase)
        if len(self.tokens) < self.seq_len:
            self._seq_len = len(self.tokens)
        return tokenizer

    def execute(self, r, tokenizer: Optional[TokenizerAbstract]=None) -> Optional[set]:
        if not hasattr(self, 'tokens'):
            self.tokenize(tokenizer)
        if len(self.tokens) == 0:
            logging.warning("No tokens found. Phrase might have been eaten by tokeinzer.")
            return None
        assert self._seq_len <= len(self.tokens), "Sequence length cannot be greater than number of tokens"
        tokens = [f"token:{t}" for t in self.tokens if t]
        results = self._get_positions(r, tokens)

        if len(results) < self._seq_len:
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
            for j in range(1, self._seq_len):
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

    def execute(self, r, tokenizer: Optional[TokenizerAbstract]=None) -> set:
        result = set()
        if tokenizer is None:
            tokenizer = get_tokenizer()
        for child in self.children:
            result = result.union(child.execute(r, tokenizer))
        print(f"OR Operator result: {result}")
        return result

class ANDOperator(AbstractOperator):

    def __init__(self, children: List[AbstractOperator]):
        self.children = self._simplify_query(children)

    def execute(self, r, tokenizer: Optional[TokenizerAbstract]=None) -> Optional[set]:
        intersection = set()
        if tokenizer is None:
            tokenizer = get_tokenizer()
        for child in self.children:
            if len(intersection) == 0:
                result = child.execute(r, tokenizer)
                if result is not None:
                    intersection = result
            else:
                result = child.execute(r, tokenizer)
                if result is not None:
                    intersection.intersection_update(result)
        print(f"AND Operator result: {intersection}")
        return intersection
    
    
    def _simplify_query(self, children: Sequence[AbstractOperator]) -> Sequence[AbstractOperator]:
        term = ""
        simplified_children = []
        for child in children:
            if isinstance(child, TermOperator):
                term  += child.phrase + " "
            else:
                simplified_children.append(child)
        if term != "":
            combined_term = TermOperator(term)
            simplified_children = [combined_term] + simplified_children
        return simplified_children

class ANDNOTOperator(AbstractOperator):

    def __init__(self, children: List[AbstractOperator]):
        self.children = children
        assert len(children) == 2, "AndNotOperator requires exactly two children"

    def execute(self, r, tokenizer: Optional[TokenizerAbstract]=None) -> set:
        if tokenizer is None:
            tokenizer = get_tokenizer()
        
        result: set = self.children[0].execute(r, tokenizer) or set()
        result.difference_update(self.children[1].execute(r, tokenizer) or set())
        print(f"ANDNOT Operator result: {result}")
        return result
    