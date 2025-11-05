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

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        raw_tokens = self._word_re.findall(text)
        return self._post_process_tokens(raw_tokens, self.config, self.stopwords)


