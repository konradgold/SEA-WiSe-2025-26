from abc import abstractmethod
import re
from typing import Any, Optional
import unicodedata
from dataclasses import dataclass
from .stopwords import get_default_stopwords
from sea.utils.config import Config


class TokenizerAbstract:
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass
    
    def _post_process_tokens(
        self, tokens: list[str], config: "TokenizerConfig", stopwords: set[str]
    ) -> list[str]:
        """Applies normalization, stemming, stopword removal, and length filtering."""
        out: list[str] = []
        for t in tokens:
            s = normalize_token(t, config)
            if not s:
                continue
            if config.stemming:
                s = simple_stem(s)
            if config.remove_stopwords and s in stopwords:
                continue
            if len(s) < config.min_len:
                continue
            out.append(s)
        return out


@dataclass
class TokenizerConfig:
    lowercase: bool = True
    ascii_fold: bool = True
    min_len: int = 2
    remove_stopwords: bool = True
    stemming: bool = False
    number_normalize: bool = True


def _env_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, str):
        try:
            return bool(value)
        except Exception:
            return default
    v = value.strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def normalize_token(token: str, cfg: TokenizerConfig) -> str:
    s = token
    if cfg.lowercase:
        s = s.lower()
    if cfg.ascii_fold:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    if cfg.number_normalize:
        if any(c.isdigit() for c in s) and "," in s:
            s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s


def simple_stem(token: str) -> str:
    s = token
    for suf in ("ing", "edly", "ed", "ly", "es", "s"):
        if len(s) > 3 and s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s


def get_tokenizer(cfg: Optional[Config]=None) -> TokenizerAbstract:
    from .simple_tokenizer import SimpleTokenizer

    config_yaml = cfg if cfg is not None else Config()
    tkcfg = TokenizerConfig(
        lowercase=_env_bool(config_yaml.TOKENIZER.LOWERCASE, True),
        ascii_fold=_env_bool(config_yaml.TOKENIZER.ASCII_FOLD, True),
        min_len=int(config_yaml.TOKENIZER.MIN_LEN),
        remove_stopwords=_env_bool(config_yaml.TOKENIZER.REMOVE_STOPWORDS, True),
        stemming=_env_bool(config_yaml.TOKENIZER.STEM, False),
        number_normalize=_env_bool(config_yaml.TOKENIZER.NUMBER_NORMALIZE, True),
    )
    stop = get_default_stopwords() if tkcfg.remove_stopwords else set()

    return SimpleTokenizer(config=tkcfg, stopwords=stop)


def __getattr__(name: str):
    """Get a tokenizer by name (backwards compatibility like from tokenization import SimpleTokenizer)"""
    if name == "SimpleTokenizer":
        from .simple_tokenizer import SimpleTokenizer
        return SimpleTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
