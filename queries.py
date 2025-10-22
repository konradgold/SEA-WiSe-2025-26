from enum import Enum
from typing import List


class AbstractOperator:
    def execute(self) -> dict:
        raise NotImplementedError
    
    @staticmethod
    def is_terminal() -> bool:
        return False
    

class TermOperator(AbstractOperator):
    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def exectute(self) -> dict:
        '''
        This just searches for the phrase
        '''
        pass

    @staticmethod
    def is_terminal() -> bool:
        return True

class PhraseOperator(AbstractOperator):
    def __init__(self, phrase: str):
        self.phrase = phrase
    
    def exectute(self) -> dict:
        '''
        This just searches for the phrase
        '''
        # Implementation of phrase search
        pass

    @staticmethod
    def is_terminal() -> bool:
        return True

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
    
    @staticmethod
    def is_terminal() -> bool:
        return False    

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
    
    @staticmethod
    def is_terminal() -> bool:
        return False

class ANDNOTOperator(AbstractOperator):
    def __init__(self, children: List[AbstractOperator]):
        self.children = children
        assert len(children) == 2, "AndNotOperator requires exactly two children" 
    
    def execute(self) -> set:
        result:set = self.children[0].exectute()
        result.difference_update(self.children[1].exectute())
        return result
    
    @staticmethod
    def is_terminal(cls) -> bool:
        return False
    



# Identify operators based on their string representation
# [is_terminal] indicates if a operator is atomic / a leaf in the query tree
class Operators(Enum):
    AND = {
        "identifiers": ["and", "&", "&&"],
        "operator": ANDOperator
    }
    ANDNOT = {
        "identifiers": ["andnot", "and not", "-"],
        "operator": ANDNOTOperator
    }
    OR = {  
        "identifiers": ["or", "|", "||"],
        "operator": OROperator
    }
    TERM = {
        "identifiers": [],
        "operator": TermOperator
    }


    @classmethod
    def get_operator_for_identifier(cls, identifier: str):
        identifier = identifier.lower()
        for operator in cls:
            if identifier in operator.value["identifiers"]:
                return operator.value["operator"]
        return Operators.TERM.value["operator"]  # Default to TERM operator


class QueryEngine:

    def _process_phrase2query(self, phrase : str) -> AbstractOperator:
        '''
        Process the query and return the root operator
        '''
        phrase = self._preprocess_phrase(phrase)
        phrase_elements = phrase.split(" ")

        # identify operator classes for each element
        operator_class = []
        for element in phrase_elements:
            operator_class.append(Operators.get_operator_for_identifier(element))

        # construct tree
        # 1. identify OR operators and their children
        # 2. identify AND & ANDNOT operators and their children     
        idx = 0 

        parse_order = [[Operators.OR.value["operator"]],
                        [Operators.AND.value["operator"], Operators.ANDNOT.value["operator"]]],
        
        for current_operator in parse_order:
            while len(phrase_elements) > idx:
                element = phrase_elements[idx]
                clazz = operator_class[idx]
                if clazz in current_operator:
                    preTerm = idx - 1
                    postTerm = idx + 1
                    if preTerm >= 0 and postTerm < len(phrase_elements):
                        if operator_class[preTerm].is_terminal() and operator_class[postTerm].is_terminal():
                            # create PhraseOperators for both terminals
                            left_child = operator_class[preTerm](phrase_elements[preTerm])
                            right_child = operator_class[postTerm](phrase_elements[postTerm])
                            # create OROperator and replace in phrase_elements
                            phrase_elements[idx] = clazz([left_child, right_child])
                            # pop the used elements
                            phrase_elements.pop(postTerm)
                            phrase_elements.pop(preTerm)
                            operator_class.pop(postTerm)
                            operator_class.pop(preTerm)
                idx += 1



        return phrase_elements[0]



        


    def _preprocess_phrase(self, query : str) -> str:
        '''
        Preprocess the query string
        '''
        query = query.lower()
        return query
        



class QueryParser:
    ...

class QueryEvaluator:
    ...


