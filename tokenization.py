import re
import unicodedata
import os


class TokenizerAbstract:
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Subclasses should implement this method.")


class SimpleTokenizer(TokenizerAbstract):
    _word_re = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

    def __init__(self, lowercase: bool = True, ascii_fold: bool = True):
        self.lowercase = lowercase
        self.ascii_fold = ascii_fold

    def _normalize(self, text: str) -> str:
        s = text
        if self.lowercase:
            s = s.lower()
        if self.ascii_fold:
            s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
        return s

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        s = self._normalize(text)
        return self._word_re.findall(s)


class SpacyTokenizer(TokenizerAbstract):
    _word_re = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

    def __init__(self, model: str | None = None, disable: list[str] | None = None, lowercase: bool = True, ascii_fold: bool = True):
        self.lowercase = lowercase
        self.ascii_fold = ascii_fold
        try:
            import spacy
            if model in (None, "", "blank"):
                self.nlp = spacy.blank("en")
            else:
                self.nlp = spacy.load(model, disable=disable or [
                    "tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "textcat"
                ])
        except Exception:
            # Fallback to a blank English tokenizer if model/unavailable
            import spacy
            self.nlp = spacy.blank("en")

    def _normalize(self, text: str) -> str:
        s = text
        if self.lowercase:
            s = s.lower()
        if self.ascii_fold:
            s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
        return s

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        # Use only the tokenizer for speed
        doc = self.nlp.make_doc(text)
        tokens: list[str] = []
        for t in doc:
            s = self._normalize(t.text)
            if not s:
                continue
            # Prefer full-match to keep spaCy token boundaries; fallback to split to salvage alnum substrings
            if self._word_re.fullmatch(s):
                tokens.append(s)
            else:
                tokens.extend(self._word_re.findall(s))
        return tokens


def _default_backend() -> str:
    env_backend = os.getenv("TOKENIZER_BACKEND")
    if env_backend:
        return env_backend.lower()
    # Prefer spaCy if available
    try:
        import spacy  # noqa: F401
        return "spacy"
    except Exception:
        return "simple"


def get_tokenizer(backend: str | None = None) -> TokenizerAbstract:
    backend = (backend or _default_backend()).lower()
    if backend == "spacy":
        model = os.getenv("SPACY_MODEL", "blank")
        disable = tuple(filter(None, (os.getenv("SPACY_DISABLE", "").split(",") if os.getenv("SPACY_DISABLE") else [])))
        return SpacyTokenizer(model=model, disable=list(disable) if disable else None)
    return SimpleTokenizer()


