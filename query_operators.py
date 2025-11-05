
from typing import List, Optional

from requests import get

from tokenization import get_tokenizer


class AbstractOperator:
    def execute(self, r) -> set:
        """
        r: Redis connection: Should not be openend/closed here, try avoid dos attack ;)
        """
        raise NotImplementedError

class TermOperator(AbstractOperator):
    def __init__(self, phrase: str):
        self.phrase = phrase
        self.tokenizer = get_tokenizer()
    
    def execute(self, r) -> set:
        tokens = self.tokenizer.tokenize(self.phrase)
        tokens = [f"token:{t}" for t in tokens if t]  # filter out empty tokens
        result = set()
        for token in tokens:
            inv_result = r.hgetall(token)
            if inv_result:
                inv_result = set(key.decode() for key, _ in inv_result.items())
            result = result.union(inv_result)
        return result

class PhraseOperator(AbstractOperator):

    seq_len: Optional[int] = None

    def __init__(self, phrase: str, ):
        self.phrase = phrase
        self.tokenizer = get_tokenizer()
    
    def execute(self, r) -> set:
        if self.seq_len is None:
            self.seq_len = 2
        tokens = self.tokenizer.tokenize(self.phrase)
        tokens = [f"token:{t}" for t in tokens if t]  #
        results = []
        for token in tokens:
            inv_result = r.hgetall(token)
            if inv_result:
                inv_result = {key.decode(): value.decode() for key, value in inv_result.items()}
            results.append(inv_result)
        # Find documents where tokens appear in consecutive positions
        matching_docs = set()
        if len(results) < self.seq_len:
            return matching_docs
            
        # Create sliding windows
        for i in range(len(results) - self.seq_len + 1):
            window = results[i:i + self.seq_len]
            
            # Find common document IDs across all dicts in window
            if not window:
                continue
            common_keys = set(window[0].keys())
            for result_dict in window[1:]:
                common_keys.intersection_update(result_dict.keys())
            
            # Check if positions are consecutive for each common document
            for doc_id in common_keys:
                positions_lists = [result_dict[doc_id] for result_dict in window]
                

                positions = []
                for pos_list in positions_lists:
                    positions.append([int(pos_list)])
                
                # Check if there's a consecutive sequence
                for pos in positions[0]:
                    is_consecutive = True
                    for j in range(1, len(positions)):
                        if (pos + j) not in positions[j]:
                            is_consecutive = False
                            break
                    if is_consecutive:
                        matching_docs.add(doc_id)
                        break

        return matching_docs
        # TODO: Implement phrase search
        pass

class OROperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children 

    def execute(self, r) -> set:
        '''
        Joins result 
        '''
        result = set()
        for child in self.children:
            result = result.union(child.execute(r))
        print(f"OR Operator result: {result}")
        return result     

class ANDOperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children 

    def execute(self, r) -> set:
        result = set()
        for child in self.children:
            if len(result) == 0:
                result = child.execute(r)
            else:
                result.intersection_update(child.execute(r))
        print(f"AND Operator result: {result}")
        return result

class ANDNOTOperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children
        assert len(children) == 2, "AndNotOperator requires exactly two children" 
    
    def execute(self, r) -> set:
        result:set = self.children[0].execute(r)
        result.difference_update(self.children[1].execute(r))
        print(f"ANDNOT Operator result: {result}")
        return result
   
