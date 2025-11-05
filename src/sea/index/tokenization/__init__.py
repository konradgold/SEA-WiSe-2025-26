import re
import unicodedata
import os
from dataclasses import dataclass
from .stopwords import get_default_stopwords


class TokenizerAbstract:
    """Abstract base class for tokenizers."""
    
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Subclasses should implement this method.")
    
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


def _default_backend() -> str:
    env_backend = os.getenv("TOKENIZER_BACKEND")
    if env_backend:
        return env_backend.lower()
    try:
        import spacy  # noqa: F401
        return "spacy"
    except Exception:
        return "simple"


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
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


class _ConfiguredTokenizer(TokenizerAbstract):
    def __init__(
        self, backend: TokenizerAbstract, cfg: TokenizerConfig, stopwords: set[str]
    ):
        self.backend = backend
        self.cfg = cfg
        self.stopwords = stopwords

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        raw_tokens = self.backend.tokenize(text)
        return self._post_process_tokens(raw_tokens, self.cfg, self.stopwords)


def get_tokenizer() -> TokenizerAbstract:
    from .simple_tokenizer import SimpleTokenizer
    from .spacy_tokenizer import SpacyTokenizer
    
    backend_name = _default_backend()
    cfg = TokenizerConfig(
        lowercase=_env_bool("TOKENIZER_LOWERCASE", True),
        ascii_fold=_env_bool("TOKENIZER_ASCII_FOLD", True),
        min_len=int(os.getenv("TOKENIZER_MIN_LEN", "2")),
        remove_stopwords=_env_bool("TOKENIZER_REMOVE_STOPWORDS", True),
        stemming=_env_bool("TOKENIZER_STEM", False),
        number_normalize=_env_bool("TOKENIZER_NUMBER_NORMALIZE", True),
    )
    stop = get_default_stopwords() if cfg.remove_stopwords else set()

    if backend_name == "spacy":
        model = os.getenv("SPACY_MODEL", "blank")
        disable = tuple(filter(None, (os.getenv("SPACY_DISABLE", "").split(",") if os.getenv("SPACY_DISABLE") else [])))
        backend = SpacyTokenizer(
            model=model, disable=list(disable) if disable else None
        )
        return _ConfiguredTokenizer(backend=backend, cfg=cfg, stopwords=stop)
    return SimpleTokenizer(config=cfg, stopwords=stop)


def __getattr__(name: str):
    """Get a tokenizer by name (backwards compatibility like from tokenization import SimpleTokenizer)"""
    if name == "SimpleTokenizer":
        from .simple_tokenizer import SimpleTokenizer
        return SimpleTokenizer
    if name == "SpacyTokenizer":
        from .spacy_tokenizer import SpacyTokenizer
        return SpacyTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


