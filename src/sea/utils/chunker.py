from collections import Counter
from typing import List, Optional

from numpy import mean
from omegaconf import DictConfig
from torch import chunk

from sea.utils.config_wrapper import Config


class Chunker:
    def __init__(self, cfg: Optional[DictConfig] = None) -> None:
        if cfg is None:
            cfg = Config(load=True)
        self.max_chunk_size = cfg.CHUNKER.MAX_CHUNK_SIZE
        self.min_chunk_size = cfg.CHUNKER.MIN_CHUNK_SIZE
        self.chunk_overlap = cfg.CHUNKER.CHUNK_OVERLAP
        self.enable = cfg.CHUNKER.ENABLE
        self.query: list[str] = []

    def set_query(self, query: List[str]) -> None:
        self.query = query
    
    def chunk_text(self, text: str) -> str:
        """Return a single chunk that covers as many query terms as possible."""
        if not self.enable or not text:
            return text

        words = text.lower().split()
        words_normal = text.split()

        if not words:
            return text

        # Adjust chunk size within configured bounds.
        mean_lengths: int = int(mean([len(word) for word in words])) * 2
        chunk_size= 250 // mean_lengths # Approximate number of words to fit in 250 chars.
        step = max(1, chunk_size - self.chunk_overlap)

        if not self.query:
            return " ".join(words[:chunk_size])

        query_counter = Counter(self.query)
        query_terms = set(query_counter)

        best_score = (-1.0, -1.0, -1.0)  # (unique_hits, coverage_ratio, total_hits, -start)
        best_range = (0, min(chunk_size, len(words)))

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            score = self.calculate_score(words[start:end], query_counter)
            
            
            if score > best_score:
                best_score = score
                best_range = (start, end)

        # Extract the best chunk with original casing
        chunk_words = words_normal[best_range[0]:best_range[1]]
        marked_chunk = self._mark_query_terms(chunk_words)
        best_chunk = "..." + " ".join(marked_chunk) + "..."
        return best_chunk if best_chunk else text
    
    def calculate_score(self, chunk_words: list[str], query_counter: Counter) -> tuple[float, float, float]:
        window_counter = Counter(chunk_words)

        unique_hits = sum(1 for term in query_counter.keys() if window_counter[term] > 0)
        total_hits = sum(window_counter[term] for term, _ in query_counter.items())
        coverage_ratio = unique_hits / len(query_counter) if query_counter else 0.0

        # Primary: cover most unique terms, secondary: repeat hits, tertiary: earlier chunk.
        return (unique_hits, coverage_ratio, total_hits)

    def _mark_query_terms(self, words: list[str]) -> list[str]:
        """Mark query terms in the chunk using **term** markdown-style formatting."""
        if not self.query:
            return words
        
        marked_words = []
        for word in words:
            if word.lower().strip("`´.,!?;\"'()") in self.query:
                marked_words.append(f"**{word}**")
            else:
                marked_words.append(word)
        return marked_words

