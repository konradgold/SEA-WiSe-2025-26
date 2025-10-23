from query_operators import AbstractOperator
from query_parser import Operators


class QueryEngine:
    def _process_phrase2query(self, phrase : str) -> AbstractOperator:
        '''
        Process the query and return the root operator
        '''
        phrase_elements = phrase.lower().split(" ")

        parsing_order = [Operators.TERM, Operators.AND, Operators.ANDNOT,  Operators.OR]   
        for current_operator in parsing_order:
            idx = 0
            while len(phrase_elements) > idx:
                element = phrase_elements[idx]
                if isinstance(element, AbstractOperator):
                    idx += 1
                    continue
                clazz = Operators.get_EnumOperator(element)
                if clazz == current_operator:
                    phrase_elements = clazz(idx, phrase_elements)              
                idx += 1
        return phrase_elements[0]