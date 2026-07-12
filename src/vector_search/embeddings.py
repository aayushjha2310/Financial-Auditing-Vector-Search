"""Document embedding utilities for vector search."""

from __future__ import annotations

import hashlib

import numpy as np


def _deterministic_embed(text: str, dim: int = 384) -> np.ndarray:
    """Generate deterministic embedding from text (fallback without sentence-transformers)."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


class DocumentEmbedder:
    """Embed corporate financial documents for vector indexing."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self._use_transformer = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._use_transformer = True
        except Exception:
            print("[Embedder] sentence-transformers unavailable, using deterministic fallback.")
            self._use_transformer = False

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Embed a list of document texts into vectors."""
        if self._use_transformer and self._model:
            vectors = self._model.encode(documents, show_progress_bar=False)
            return vectors.astype(np.float32)
        return np.array([_deterministic_embed(doc, self.dim) for doc in documents], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]
