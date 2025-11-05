from enum import Enum
from typing import NamedTuple

from sea.query.operators import ANDNOTOperator, ANDOperator, AbstractOperator, OROperator, TermOperator, PhraseOperator

class OperatorSpec(NamedTuple):
    identifiers: tuple[str, ...]
    operator_cls: type
    is_terminal: bool

# Identify operators based on their string representation
# [is_terminal] indicates if a operator is atomic / a leaf in the query tree
class Operators(Enum):
    BRACKET = OperatorSpec(("("), ANDOperator,    False)
    AND    = OperatorSpec(("and", "&", "&&"), ANDOperator,    False)
    ANDNOT = OperatorSpec(("andnot", "-"), ANDNOTOperator, False)
    OR     = OperatorSpec(("or", "|", "||"), OROperator,      False)
    # has no identifiers, as any unrecognized token is treated as a TERM
    TERM   = OperatorSpec((),                 TermOperator,    True)
    PHRASE = OperatorSpec(("'",),                 PhraseOperator,  True)

    def __init__(self, identifiers, operator_cls, is_terminal):
        self.identifiers   = identifiers
        self.operator_cls  = operator_cls
        self.is_terminal   = is_terminal

    @classmethod
    def get_EnumOperator(cls, identifier: str) -> "Operators":
        id_ = identifier.lower()
        for member in cls: 
            if id_ in member.identifiers:
                return member
            if member == cls.PHRASE and id_.startswith("'"):
                return member
            if member == cls.BRACKET and id_.startswith("("):
                return member
        return cls.TERM

    @classmethod
    def get_AbstractOperator(cls, identifier: str) -> AbstractOperator:
        return cls.get_EnumOperator(identifier).operator_cls