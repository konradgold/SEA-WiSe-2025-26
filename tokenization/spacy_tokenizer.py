"""Spacy-based tokenizer that uses spaCy's tokenization pipeline."""
import re
from . import TokenizerAbstract


class SpacyTokenizer(TokenizerAbstract):
    """
    Tokenizer that uses spaCy's tokenization pipeline.

    Falls back to regex-based extraction if spaCy tokenization produces
    tokens that don't match the expected alphanumeric pattern.
    """
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


