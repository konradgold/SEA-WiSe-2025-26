
from typing import List


class AbstractOperator:
    def execute(self) -> set:
        raise NotImplementedError

class TermOperator(AbstractOperator):
    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def execute(self) -> set:
        '''
        This just searches for the phrase
        '''
        print(f"Searching for term: {self.phrase}")
        return {f"ID-{self.phrase}"}

class PhraseOperator(AbstractOperator):
    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def execute(self) -> set:
        '''
        This just searches for the phrase
        '''
        # Implementation of phrase search
        pass

class OROperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children 

    def execute(self) -> set:
        '''
        Joins result 
        '''
        result = set()
        for child in self.children:
            result = result.union(child.execute())
        print(f"OR Operator result: {result}")
        return result     

class ANDOperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children 

    def execute(self) -> set:
        result = set()
        for child in self.children:
            if len(result) == 0:
                result = child.execute()
            else:
                result.intersection_update(child.execute())
        print(f"AND Operator result: {result}")
        return result

class ANDNOTOperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children
        assert len(children) == 2, "AndNotOperator requires exactly two children" 
    
    def execute(self) -> set:
        result:set = self.children[0].execute()
        result.difference_update(self.children[1].execute())
        print(f"ANDNOT Operator result: {result}")
        return result
   
