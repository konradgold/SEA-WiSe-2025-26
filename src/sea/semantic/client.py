"""HTTP client for the embedding service."""

from typing import Literal

import httpx
import numpy as np


class EmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _embed(self, texts: list[str], text_type: Literal["query", "document"]) -> np.ndarray:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/embed",
                json={"texts": texts, "type": text_type},
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["embeddings"], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns shape (dim,)."""
        return self._embed([text], "query")[0]

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed multiple queries. Returns shape (n, dim)."""
        return self._embed(texts, "query")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents. Returns shape (n, dim)."""
        return self._embed(texts, "document")

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
