from sea.query.operators import ANDOperator, AbstractOperator, PhraseOperator
from sea.query.specs import Operators

from typing import Callable, List


class QueryParser:

    operator2parser : dict[Operators, Callable[[int, List [str | AbstractOperator]], List [str | AbstractOperator]]]

    def __init__(self, cfg):
        self.operator2parser = {
            Operators.BRACKET : self.parse_bracket,
            Operators.AND : self.parse_pre_and_post,
            Operators.OR : self.parse_pre_and_post,
            Operators.ANDNOT : self.parse_pre_and_post,
            Operators.PHRASE : self.parse_phrase,
            Operators.TERM : self.parse_self}
        self.cfg = cfg
    

    def process_phrase2query(self, phrase : str) -> AbstractOperator:
        '''
        Process the query and return the root operator
        '''
        phrase_elements = phrase.lower().split(" ")
        phrase_elements = [elem for elem in phrase_elements if elem != ''] # remove empty elements
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
                    elements = self.operator2parser[clazz](idx, elements)   
                idx += 1

        if (len(elements) != 1) :
            return ANDOperator(elements)
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
        idx_end = idx_start + 1
        for idx_end in range(idx_start + 1, len(query_elements)):
            element = query_elements[idx_end]
            if isinstance(element, str) and "'" in element:
                break
        phrase = " ".join(query_elements[idx_start:idx_end+1])
        query_element = PhraseOperator(phrase)
        query_element.seq_len = self.cfg.QUERY.MAX_PHRASE_LEN
        query_elements[idx_start] = query_element
        # remove used elements
        for _ in range(idx_end - idx_start):
            query_elements.pop(idx_start + 1)
        return query_elements