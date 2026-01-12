from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch


def detect_device(requested: str) -> str:
    """Auto-detect the best available device for PyTorch.

    Args:
        requested: Device from config ("auto", "cuda", "cuda:0", "mps", "cpu")

    Returns:
        Validated device string that is actually available.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # Validate requested device is available
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        # Check specific device index if provided (e.g., "cuda:1")
        if ":" in requested:
            device_idx = int(requested.split(":")[1])
            if device_idx >= torch.cuda.device_count():
                print(f"Warning: {requested} not available (only {torch.cuda.device_count()} GPUs). Using cuda:0.")
                return "cuda:0"
        return requested

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print(f"Warning: MPS requested but not available. Falling back to CPU.")
            return "cpu"
        return requested

    return requested  # cpu or unknown


class SpladeEncoder:

    def __init__(self, cfg=None):
        self.cfg = cfg
        model_id = cfg.SPLADE.MODEL_ID if cfg else 'naver/splade-cocondenser-ensembledistil'
        self.cutoff = cfg.SPLADE.CAP_EXPANSION if cfg else 3
        cache_dir = cfg.SPLADE.CACHE_DIR if cfg else None
        requested_device = cfg.SPLADE.DEVICE if cfg else "auto"
        self.device = detect_device(requested_device)
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

    def _encode(self, text: str):
        tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
        take_top = tokens["input_ids"].shape[1] * self.cutoff

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
        sparse_dict_tokens=  [
            (k, v) for k, v in sorted(
                sparse_dict_tokens.items(),
                key=lambda item: item[1],
                reverse=True
            )
            if v >= self.threshold
        ][:take_top]
        return sparse_dict, sparse_dict_tokens
    
    def expand(self, text: str) -> list[str]:
        _, sparse_dict_tokens = self._encode(text)
        tokens = set(k for k, v in sparse_dict_tokens)
        input_tokens = set(self.tokenizer.tokenize(text))
        tokens -= input_tokens
        tokens = list(tokens)[:self.cutoff]
        out = tokens + text.split()
        return out
    
    def tokenize(self, text):
        return list(self.tokenizer.tokenize(text))
