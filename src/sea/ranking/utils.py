from functools import total_ordering
from typing import Callable, Optional
from rich.console import Console
from rich.markdown import Markdown
from sea.utils.chunker import Chunker

console = Console()


@total_ordering
class Document:
    def __init__(self, doc_id: str, link: str, title: str, content: Optional[str], score: float = 0.0):
        self.doc_id = doc_id
        self.link = link
        self.title = title
        self.content = content
        self.score = score

    def pprint(self, verbose: bool = False, loud: bool = False, chunker: Optional[Chunker]=None, rank: Optional[int] = None) -> str:
        rank_prefix = f"#{rank}  " if rank is not None else ""
        if verbose:
            if chunker is not None and self.content is not None:
                self.content = chunker.chunk_text(self.content)
            t = f"{rank_prefix}**{self.title}**\n\n{self.link}\n\n{self.content}\n\nScore: {self.score:.2f}"
        else:
            t = f"{rank_prefix}{self.title}\n{self.link}\nScore: {self.score:.2f}"
        if loud:
            console.print(Markdown(t))
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
    
