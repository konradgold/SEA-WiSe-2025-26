"""Device auto-detection for PyTorch."""

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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        if ":" in requested:
            device_idx = int(requested.split(":")[1])
            if device_idx >= torch.cuda.device_count():
                print(f"Warning: {requested} not available (only {torch.cuda.device_count()} GPUs). Using cuda:0.")
                return "cuda:0"
        return requested

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return "cpu"
        return requested

    return requested
