from enum import Enum
from typing import Callable, List, NamedTuple

from query_operators import ANDNOTOperator, ANDOperator, AbstractOperator, OROperator, TermOperator


class OperatorSpec(NamedTuple):
    identifiers: tuple[str, ...]
    operator_cls: type
    is_terminal: bool
    parsing_strategy: Callable[[int, List [str | AbstractOperator]], List [str | AbstractOperator]]    

# Strategy for AND, OR, ANDNOT operators
def parse_pre_and_post( idx: int, query_elements: List[str | AbstractOperator]) -> List [str | AbstractOperator]:
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
def parse_self(idx: int, query_elements: List[str | AbstractOperator]) -> List [str | AbstractOperator]:
    clazz = Operators.get_AbstractOperator(query_elements[idx])
    query_elements[idx] = clazz(query_elements[idx])
    return query_elements

# Identify operators based on their string representation
# [is_terminal] indicates if a operator is atomic / a leaf in the query tree
class Operators(Enum):
    AND    = OperatorSpec(("and", "&", "&&"), ANDOperator,    False, parse_pre_and_post)
    ANDNOT = OperatorSpec(("andnot", "-"), ANDNOTOperator, False, parse_pre_and_post)
    OR     = OperatorSpec(("or", "|", "||"), OROperator,      False, parse_pre_and_post)
    # has no identifiers, as any unrecognized token is treated as a TERM
    TERM   = OperatorSpec((),                 TermOperator,    True,  parse_self)

    def __init__(self, identifiers, operator_cls, is_terminal, parsing_strategy):
        self.identifiers   = identifiers
        self.operator_cls  = operator_cls
        self.is_terminal   = is_terminal
        self._process_func = parsing_strategy
        
    def __call__(self, idx: int, query_elements: List[str | AbstractOperator]) -> List [str | AbstractOperator]:
        return self._process_func(idx, query_elements)

    @classmethod
    def get_EnumOperator(cls, identifier: str) -> Operators:
        id_ = identifier.lower()
        for member in cls:
            if id_ in member.identifiers:
                return member
        return cls.TERM

    @classmethod
    def get_AbstractOperator(cls, identifier: str) -> AbstractOperator:
        return cls.get_EnumOperator(identifier).operator_cls
