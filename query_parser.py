from query_operators import AbstractOperator, PhraseOperator
from query_operator_specs import Operators

from typing import Callable, List


class QueryEngine:

    operator2parser : dict[Operators, Callable[[QueryEngine,int, List [str | AbstractOperator]], List [str | AbstractOperator]]]

    def __init__(self):
        self.operator2parser = {
            Operators.BRACKET : QueryEngine.parse_bracket,
            Operators.AND : QueryEngine.parse_pre_and_post,
            Operators.OR : QueryEngine.parse_pre_and_post,
            Operators.ANDNOT : QueryEngine.parse_pre_and_post,
            Operators.PHRASE : QueryEngine.parse_phrase,
            Operators.TERM : QueryEngine.parse_self}
    

    def process_phrase2query(self, phrase : str) -> AbstractOperator:
        '''
        Process the query and return the root operator
        '''
        phrase_elements = phrase.lower().split(" ")
        return self._process_phraseElements(phrase_elements)

    def _process_phraseElements(self, elements : List[str | AbstractOperator]):

        parsing_order = [Operators.PHRASE, Operators.BRACKET,  Operators.TERM, Operators.AND, Operators.ANDNOT,  Operators.OR]   
        for current_operator in parsing_order:
            idx = 0
            while len(elements) > idx:
                element = elements[idx]
                if isinstance(element, AbstractOperator):
                    idx += 1
                    continue
                clazz = Operators.get_EnumOperator(element)
                if clazz == current_operator:
                    # elements = clazz(self, idx, elements)           
                    elements = self.operator2parser[clazz](self, idx, elements)   
                idx += 1
        return elements[0]
    
    
    # Strategy for AND, OR, ANDNOT operators
    def parse_pre_and_post(self, idx: int, query_elements: List[str | AbstractOperator]) -> List [str | AbstractOperator]:
        clazz = Operators.get_AbstractOperator(query_elements[idx])
        preTerm = idx - 1
        postTerm = idx + 1
        if preTerm >= 0 and postTerm < len(query_elements):
            leftChild = query_elements[preTerm]
            rightChild = query_elements[postTerm]
            query_elements[idx] = clazz([leftChild, rightChild])
            # pop the used elements
            query_elements.pop(postTerm)
            query_elements.pop(preTerm)
        return query_elements

    # Strategy for TERM operator
    def parse_self(self, idx: int, query_elements: List[str | AbstractOperator]) -> List [str | AbstractOperator]:
        clazz = Operators.get_AbstractOperator(query_elements[idx])
        query_elements[idx] = clazz(query_elements[idx])
        return query_elements

    def parse_bracket(self, idx_start: int, query_elements: List[str | AbstractOperator]) -> List [str | AbstractOperator]:
        # necessary to handle nested brackets
        bracketDepth = 0
        for idx_end in range(idx_start + 1, len(query_elements)):
                element = query_elements[idx_end]
                if isinstance(element, str) and "(" in element:
                    bracketDepth += 1
                if isinstance(element, str) and ")" in element:
                    if bracketDepth > 0:
                        bracketDepth -= 1
                    else:
                        break
        bracket = query_elements[idx_start:idx_end+1]
        bracket[0] = bracket[0].replace("(", "")
        bracket[-1] = bracket[-1].replace(")", "")
        query_elements[idx_start] = self._process_phraseElements(bracket)
        # remove used elements
        for _ in range(idx_end - idx_start):
            query_elements.pop(idx_start + 1)
        return query_elements

    def parse_phrase(self, idx_start: int, query_elements: List[str | AbstractOperator]) -> List [str | AbstractOperator]:
        for idx_end in range(idx_start + 1, len(query_elements)):
            element = query_elements[idx_end]
            if isinstance(element, str) and "'" in element:
                break
        phrase = " ".join(query_elements[idx_start:idx_end+1])
        clazz = Operators.get_AbstractOperator(phrase)
        query_elements[idx_start] = clazz(phrase)
        # remove used elements
        for _ in range(idx_end - idx_start):
            query_elements.pop(idx_start + 1)
        return query_elements