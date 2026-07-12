"""FAISS IVFPQ vector similarity search engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FAISSVectorEngine:
    """
    Facebook AI Similarity Search with Inverted File Product Quantization (IVFPQ)
    for low-latency in-memory batch vector similarity retrieval.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        nlist: int = 64,
        m_pq: int = 8,
        nbits: int = 8,
        nprobe: int = 16,
    ):
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.m_pq = m_pq
        self.nbits = nbits
        self.nprobe = nprobe
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self._is_trained = False

    def _build_index(self) -> faiss.Index:
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            self.embedding_dim,
            self.nlist,
            self.m_pq,
            self.nbits,
        )
        index.nprobe = self.nprobe
        return index

    def train_and_build(self, vectors: np.ndarray) -> None:
        """Train IVFPQ index on vector corpus and add all vectors."""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)

        self.index = self._build_index()
        n_train = min(len(vectors), max(self.nlist * 10, 256))
        self.index.train(vectors[:n_train])
        self.index.add(vectors)
        self._is_trained = True

    def search(
        self, query_vectors: np.ndarray, top_k: int = 10
    ) -> list[list[dict[str, Any]]]:
        """Batch similarity search across indexed vectors."""
        if self.index is None or not self._is_trained:
            raise RuntimeError("Index not built. Call train_and_build first.")

        queries = np.ascontiguousarray(query_vectors.astype(np.float32))
        faiss.normalize_L2(queries)

        distances, indices = self.index.search(queries, top_k)
        results = []
        for q_idx in range(len(queries)):
            query_results = []
            for rank, (dist, idx) in enumerate(zip(distances[q_idx], indices[q_idx])):
                if idx < 0:
                    continue
                entry = {
                    "rank": rank + 1,
                    "index": int(idx),
                    "distance": float(dist),
                    "similarity": float(1.0 / (1.0 + dist)),
                }
                if idx < len(self.metadata):
                    entry["metadata"] = self.metadata[idx]
                query_results.append(entry)
            results.append(query_results)
        return results

    def set_metadata(self, metadata: list[dict[str, Any]]) -> None:
        self.metadata = metadata

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "faiss_ivfpq.index"))
        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        index_path = directory / "faiss_ivfpq.index"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.index.nprobe = self.nprobe
            self._is_trained = True
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

    @property
    def vector_count(self) -> int:
        return self.index.ntotal if self.index else 0
