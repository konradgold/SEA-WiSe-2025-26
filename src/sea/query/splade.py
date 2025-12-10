from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class SpladeEncoder:

    def __init__(self, cfg=None):
        self.cfg = cfg
        model_id = cfg.SPLADE.MODEL_ID if cfg else 'naver/splade-cocondenser-ensembledistil'
        self.cutoff = cfg.SPLADE.CAP_EXPANSION if cfg else 10
        cache_dir = cfg.SPLADE.CACHE_DIR if cfg else None
        self.device = DEVICE
        self.threshold = cfg.SPLADE.THRESHOLD if cfg else 0.0
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, device=self.device, cache_dir=cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id, cache_dir=cache_dir).to(self.device)
        self.idx2token = {
            idx: token for token, idx in self.tokenizer.get_vocab().items()
        }
    
    def to(self, device: str):
        self.device = device
        self.model.to(device)
        self.tokenizer.device = device

    def encode(self, text: str):
        tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model(**tokens)
        vec = torch.max(
            torch.log(
                1 + torch.relu(output.logits)
            ) * tokens.attention_mask.unsqueeze(-1),
            dim=1)[0].squeeze()
        cols = vec.nonzero().squeeze().cpu().tolist()
        weights = vec[cols].cpu().tolist()
        sparse_dict = dict(zip(cols, weights))

        sparse_dict_tokens = {
            self.idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
        }
        # sort so we can see most relevant tokens first
        sparse_dict_tokens = {
            k: v for k, v in sorted(
                sparse_dict_tokens.items(),
                key=lambda item: item[1],
                reverse=True
            )
            if v >= self.threshold
        }
        return sparse_dict, sparse_dict_tokens
    
    def expand(self, text: str):
        _, sparse_dict_tokens = self.encode(text)
        tokens = set(sparse_dict_tokens.keys())
        input_tokens = set(self.tokenizer.tokenize(text))
        tokens -= input_tokens
        tokens = list(tokens)[:self.cutoff]
        out = tokens + list(input_tokens)
        return out
    
    def tokenize(self, text):
        return list(self.tokenizer.tokenize(text))
