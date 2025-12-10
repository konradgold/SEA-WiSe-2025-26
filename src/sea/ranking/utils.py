from functools import total_ordering
from typing import Callable, Optional


@total_ordering
class Document:
    def __init__(self, doc_id: str, link: str, title: str, content: Optional[str], score: float = 0.0):
        self.doc_id = doc_id
        self.link = link
        self.title = title
        self.content = content
        self.score = score

    def pprint(self, verbose: bool = False, loud: bool = False) -> str:
        if verbose:
            t = f"Document ID: {self.doc_id}\nLink: {self.link}\nTitle: {self.title}\nContent: {self.content}\n\nScoring: {self.score}"
        else:
            t = f"Document ID: {self.doc_id}\nTitle: {self.title}\nScore: {self.score}"
        if loud:
            print(t)
        return t
    
    def __eq__(self, other):
        if not isinstance(other, Document): 
            return NotImplemented
        return self.score == other.score
    
    def __lt__(self, other):
        if not isinstance(other, Document): 
            return NotImplemented
        return self.score < other.score
    

class RankingRegistry:
    _registry: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, ranker_class: Callable):
        cls._registry[name] = ranker_class

    @classmethod
    def get_ranker(cls, name: str):
        ranker_class = cls._registry.get(name)
        if ranker_class is None:
            raise ValueError(f"Ranker '{name}' not found in registry.")
        return ranker_class
    
