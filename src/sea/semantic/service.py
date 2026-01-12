"""FastAPI embedding service using nomic-ai/nomic-embed-text-v1.5."""

from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from sea.utils.config_wrapper import Config

MODEL: SentenceTransformer | None = None
CFG = None
DIM = 64


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


class EmbedRequest(BaseModel):
    texts: list[str]
    type: Literal["query", "document"] = "query"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def get_model() -> SentenceTransformer:
    global MODEL
    if MODEL is None:
        raise RuntimeError("Model not loaded")
    return MODEL


def compute_embeddings(texts: list[str], text_type: str, dim: int = 64) -> np.ndarray:
    """Compute embeddings with task prefix and Matryoshka truncation."""
    model = get_model()

    prefix = "search_query: " if text_type == "query" else "search_document: "
    prefixed = [prefix + t for t in texts]

    embeddings = model.encode(prefixed, convert_to_tensor=True)

    # Matryoshka: layer_norm, truncate, L2 normalize
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :dim]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, CFG, DIM
    CFG = Config(load=True)
    model_id = getattr(CFG.SEMANTIC, "MODEL_ID", "nomic-ai/nomic-embed-text-v1.5")
    requested_device = getattr(CFG.SEMANTIC, "DEVICE", "auto")
    device = detect_device(requested_device)
    DIM = getattr(CFG.SEMANTIC, "DIM", 64)

    print(f"Loading model {model_id} on {device}...")
    MODEL = SentenceTransformer(model_id, trust_remote_code=True, device=device)
    print("Model loaded.")
    yield
    MODEL = None


app = FastAPI(title="Embedding Service", lifespan=lifespan)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    embeddings = compute_embeddings(request.texts, request.type, dim=DIM)
    return EmbedResponse(embeddings=embeddings.tolist())


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


if __name__ == "__main__":
    import uvicorn

    cfg = Config(load=True)
    port = getattr(cfg.SEMANTIC, "SERVICE_PORT", 8001)
    uvicorn.run(app, host="0.0.0.0", port=port)
