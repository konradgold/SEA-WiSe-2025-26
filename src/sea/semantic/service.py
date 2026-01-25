"""FastAPI embedding service using nomic-ai/nomic-embed-text-v1.5."""

from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from sea.utils.config_wrapper import Config
from sea.utils.device import detect_device

EMBEDDING_MODEL: SentenceTransformer | None = None
CONFIG = None
EMBEDDING_DIMENSION = 64


class EmbedRequest(BaseModel):
    texts: list[str]
    type: Literal["query", "document"] = "query"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def get_model() -> SentenceTransformer:
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        raise RuntimeError("Model not loaded")
    return EMBEDDING_MODEL


def compute_embeddings(texts: list[str], text_type: str, dimension: int = 64) -> np.ndarray:
    """Compute embeddings with task prefix and Matryoshka truncation
    """
    model = get_model()

    # Model expects specific prefixes
    task_prefix = "search_query: " if text_type == "query" else "search_document: "
    prefixed_texts = [task_prefix + text for text in texts]

    raw_embeddings = model.encode(prefixed_texts, convert_to_tensor=True)

    # Matryoshka 
    # 1. Layer norm so dims have similar scale
    # 2. Truncate to first N dimensions 
    # 3. Normalize so dot product equals cosine similarity
    normalized_embeddings = F.layer_norm(raw_embeddings, normalized_shape=(raw_embeddings.shape[1],))
    truncated_embeddings = normalized_embeddings[:, :dimension]
    unit_embeddings = F.normalize(truncated_embeddings, p=2, dim=1)

    return unit_embeddings.cpu().numpy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global EMBEDDING_MODEL, CONFIG, EMBEDDING_DIMENSION
    CONFIG = Config(load=True)
    model_id = getattr(CONFIG.SEMANTIC, "MODEL_ID", "nomic-ai/nomic-embed-text-v1.5")
    requested_device = getattr(CONFIG.SEMANTIC, "DEVICE", "auto")
    device = detect_device(requested_device)
    EMBEDDING_DIMENSION = getattr(CONFIG.SEMANTIC, "DIM", 64)

    print(f"Loading model {model_id} on {device}...")
    EMBEDDING_MODEL = SentenceTransformer(model_id, trust_remote_code=True, device=device)
    print("Model loaded.")
    yield
    EMBEDDING_MODEL = None


app = FastAPI(title="Embedding Service", lifespan=lifespan)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    embeddings = compute_embeddings(request.texts, request.type, dimension=EMBEDDING_DIMENSION)
    return EmbedResponse(embeddings=embeddings.tolist())


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": EMBEDDING_MODEL is not None}


if __name__ == "__main__":
    import uvicorn

    config = Config(load=True)
    port = getattr(config.SEMANTIC, "SERVICE_PORT", 8001)
    uvicorn.run(app, host="0.0.0.0", port=port)
