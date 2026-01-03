from sea.utils.logger import write_message_to_log_file
from transformers import AutoTokenizer
from sea.index.tokenization import TokenizerAbstract




class FastBertTokenizer(TokenizerAbstract):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True  # Ensures rust-based fast tokenizer is used
        )
        write_message_to_log_file("Initialized FastBertTokenizer with 'bert-base-uncased' model.")
        self.tokens = set()
    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        # Tokenize and convert to tokens (strings)
        tokens = self.tokenizer.tokenize(text)  # Returns [101, 2023, 2003, 1037, 3899, 102]
        self.tokens.update(tokens)
        assert len(self.tokens) < 40000, "Too many unique tokens encountered!"
        return tokens