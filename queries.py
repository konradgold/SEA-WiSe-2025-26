from typing import dict, List


class AbstractOperator:
    def execute(self, result: Optional[dict], operators: Optional[List[AbstractOperator]]) -> dict:

class TermOperator(AbstractOperator):
    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def exectute(self) -> dict:
        '''
        This just searches for the phrase
        '''
        ...

class OROperator(AbstractOperator):
    def execute(self, result, operators = Optional[List[AbstractOperator]]):
        '''
        Joins result 
        '''
        for operator in operators:
            result = operator.exectute()



class QueryParser:
    ...

class QueryEvaluator:
    ...


