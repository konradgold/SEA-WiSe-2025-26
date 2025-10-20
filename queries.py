from typing import dict, List


class AbstractOperator:
    def execute(self) -> dict:
        raise NotImplementedError
    

class TermOperator(AbstractOperator):
    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def exectute(self) -> dict:
        '''
        This just searches for the phrase
        '''
        pass

class PhraseOperator(AbstractOperator):
    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def exectute(self) -> dict:
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
            result = result.union(child.exectute())
        return result   

class ANDOperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children 

    def execute(self) -> set:
        result = set()
        for child in self.children:
            if len(result) == 0:
                result = child.exectute()
            else:
                result.intersection_update(child.exectute())
        return result

class AndNotOperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children
        assert len(children) == 2, "AndNotOperator requires exactly two children" 
    
    def execute(self) -> set:
        result:set = self.children[0].exectute()
        result.difference_update(self.children[1].exectute())
        return result
    

class QueryParser:
    ...

class QueryEvaluator:
    ...


