import re
import unicodedata
import os
from dataclasses import dataclass
from typing import Optional


class TokenizerAbstract:
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Subclasses should implement this method.")


class SimpleTokenizer(TokenizerAbstract):
    _word_re = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

    def __init__(
        self,
        lowercase: bool = True,
        ascii_fold: bool = True,
        min_len: int = 2,
        remove_stopwords: bool = True,
        stemming: bool = False,
        number_normalize: bool = True,
        stopwords: Optional[set[str]] = None,
    ):
        self.lowercase = lowercase
        self.ascii_fold = ascii_fold
        self.min_len = int(min_len)
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.number_normalize = number_normalize
        self.stopwords = (
            stopwords
            if (remove_stopwords and stopwords is not None)
            else (_default_stopwords() if remove_stopwords else set())
        )

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        raw_tokens = self._word_re.findall(text)
        out: list[str] = []
        for t in raw_tokens:
            s = _normalize_token(
                t,
                TokenizerConfig(
                    lowercase=self.lowercase,
                    ascii_fold=self.ascii_fold,
                    min_len=self.min_len,
                    remove_stopwords=self.remove_stopwords,
                    stemming=self.stemming,
                    number_normalize=self.number_normalize,
                ),
            )
            if not s:
                continue
            if self.stemming:
                s = _simple_stem(s)
            if self.remove_stopwords and s in self.stopwords:
                continue
            if len(s) < self.min_len:
                continue
            out.append(s)
        return out


class SpacyTokenizer(TokenizerAbstract):
    _word_re = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

    def __init__(self, model: str | None = None, disable: list[str] | None = None):
        try:
            import spacy
            if model in (None, "", "blank"):
                self.nlp = spacy.blank("en")
            else:
                self.nlp = spacy.load(model, disable=disable or [
                    "tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "textcat"
                ])
        except Exception:
            import spacy
            self.nlp = spacy.blank("en")

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        doc = self.nlp.make_doc(text)
        tokens: list[str] = []
        for t in doc:
            s = t.text
            if not s:
                continue
            if self._word_re.fullmatch(s):
                tokens.append(s)
            else:
                tokens.extend(self._word_re.findall(s))
        return tokens


@dataclass
class TokenizerConfig:
    lowercase: bool = True
    ascii_fold: bool = True
    min_len: int = 2
    remove_stopwords: bool = True
    stemming: bool = False
    number_normalize: bool = True  # remove commas inside numbers


def _default_stopwords() -> set[str]:
    # A compact English stopword list covering common function words
    return {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "to",
        "up",
        "down",
        "with",
        "as",
        "is",
        "it",
        "its",
        "be",
        "are",
        "was",
        "were",
        "am",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "me",
        "him",
        "her",
        "them",
        "my",
        "your",
        "his",
        "their",
        "our",
        "mine",
        "yours",
        "hers",
        "theirs",
        "ours",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "from",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "why",
        "because",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "about",
        "than",
        "too",
        "very",
        "can",
        "cannot",
        "could",
        "should",
        "would",
        "will",
        "just",
        "only",
        "own",
        "same",
        "so",
        "such",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "any",
        "all",
    }


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


def _normalize_token(token: str, cfg: TokenizerConfig) -> str:
    s = token
    if cfg.lowercase:
        s = s.lower()
    if cfg.ascii_fold:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    if cfg.number_normalize:
        # remove commas inside digit groups: 1,234 -> 1234
        if any(c.isdigit() for c in s) and "," in s:
            s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s


def _simple_stem(token: str) -> str:
    # Minimal suffix stripping to avoid extra dependencies
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
        out: list[str] = []
        for t in raw_tokens:
            s = _normalize_token(t, self.cfg)
            if not s:
                continue
            if self.cfg.stemming:
                s = _simple_stem(s)
            if self.cfg.remove_stopwords and s in self.stopwords:
                continue
            if len(s) < self.cfg.min_len:
                continue
            out.append(s)
        return out


def get_tokenizer() -> TokenizerAbstract:
    backend_name = _default_backend()
    cfg = TokenizerConfig(
        lowercase=_env_bool("TOKENIZER_LOWERCASE", True),
        ascii_fold=_env_bool("TOKENIZER_ASCII_FOLD", True),
        min_len=int(os.getenv("TOKENIZER_MIN_LEN", "2")),
        remove_stopwords=_env_bool("TOKENIZER_REMOVE_STOPWORDS", True),
        stemming=_env_bool("TOKENIZER_STEM", False),
        number_normalize=_env_bool("TOKENIZER_NUMBER_NORMALIZE", True),
    )
    stop = _default_stopwords() if cfg.remove_stopwords else set()

    if backend_name == "spacy":
        model = os.getenv("SPACY_MODEL", "blank")
        disable = tuple(filter(None, (os.getenv("SPACY_DISABLE", "").split(",") if os.getenv("SPACY_DISABLE") else [])))
        backend = SpacyTokenizer(
            model=model, disable=list(disable) if disable else None
        )
        return _ConfiguredTokenizer(backend=backend, cfg=cfg, stopwords=stop)
    # simple backend
    return SimpleTokenizer(
        lowercase=cfg.lowercase,
        ascii_fold=cfg.ascii_fold,
        min_len=cfg.min_len,
        remove_stopwords=cfg.remove_stopwords,
        stemming=cfg.stemming,
        number_normalize=cfg.number_normalize,
        stopwords=stop,
    )
