import re
from typing import Optional
from . import TokenizerAbstract, TokenizerConfig
from .stopwords import get_default_stopwords


class SimpleTokenizer(TokenizerAbstract):
    """
    Regex-based tokenizer that extracts words from text.

    The regex pattern matches:
    - [A-Za-z0-9]+ : One or more alphanumeric characters
    - (?:'[A-Za-z0-9]+)? : Optionally followed by an apostrophe and more alphanumerics

    Examples:
    - "hello world" -> ["hello", "world"]
    - "don't worry" -> ["don't", "worry"]
    - "it's 123" -> ["it's", "123"]
    """
    _word_re = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

    def __init__(
        self,
        config: TokenizerConfig | None = TokenizerConfig(),
        stopwords: Optional[set[str]] = None,
    ):
        self.config = config
        self.stopwords = (
            stopwords
            if (config.remove_stopwords and stopwords is not None)
            else (get_default_stopwords() if config.remove_stopwords else set())
        )

        # Fast path: Skip per-token processing (no ASCII folding, number normalization, or stemming)
        self._fast_path = (
            not self.config.ascii_fold
            and not self.config.number_normalize
            and not self.config.stemming
        )

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        if self._fast_path:
            if self.config.lowercase:
                text = text.lower()
            raw_tokens = self._word_re.findall(text)

            min_len = self.config.min_len
            if self.config.remove_stopwords:
                if min_len > 1:
                    return [
                        t
                        for t in raw_tokens
                        if t not in self.stopwords and len(t) >= min_len
                    ]
                return [t for t in raw_tokens if t not in self.stopwords]

            # No stopwords, but min length constraint
            if min_len > 1:
                return [t for t in raw_tokens if len(t) >= min_len]

            return raw_tokens

        # Normal path: Move lowercasing to the whole string once
        if self.config.lowercase:
            text = text.lower()
        raw_tokens = self._word_re.findall(text)
        return self._post_process_tokens(raw_tokens, self.config, self.stopwords)
