"""Pinecone managed vector database engine."""

from __future__ import annotations

from typing import Any

import numpy as np


class PineconeVectorEngine:
    """
    Fully managed Pinecone vector data management for production
    semantic verification across multi-billion vector data spaces.
    Includes mock mode for local development without API keys.
    """

    def __init__(
        self,
        index_name: str = "financial-audit-index",
        embedding_dim: int = 384,
        metric: str = "cosine",
        api_key: str = "",
        use_mock: bool = True,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.api_key = api_key
        self.use_mock = use_mock or not api_key
        self.cloud = cloud
        self.region = region
        self.index = None
        self._mock_vectors: dict[str, tuple[np.ndarray, dict]] = {}

    def initialize(self) -> bool:
        """Initialize Pinecone index (real or mock)."""
        if self.use_mock:
            print("[Pinecone] Running in mock mode (no API key required).")
            return True

        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=self.api_key)
            existing = [idx.name for idx in pc.list_indexes()]
            if self.index_name not in existing:
                pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
            self.index = pc.Index(self.index_name)
            return True
        except Exception as e:
            print(f"[Pinecone] Init failed ({e}), falling back to mock mode.")
            self.use_mock = True
            return False

    def upsert(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> int:
        """Upsert vectors with metadata into Pinecone index."""
        if ids is None:
            ids = [f"vec_{i}" for i in range(len(vectors))]

        if self.use_mock:
            for vid, vec, meta in zip(ids, vectors, metadata):
                self._mock_vectors[vid] = (vec.astype(np.float32), meta)
            return len(ids)

        records = []
        for vid, vec, meta in zip(ids, vectors, metadata):
            records.append({"id": vid, "values": vec.tolist(), "metadata": meta})

        batch_size = 100
        for i in range(0, len(records), batch_size):
            self.index.upsert(vectors=records[i : i + batch_size])
        return len(ids)

    def search(
        self, query_vectors: np.ndarray, top_k: int = 10
    ) -> list[list[dict[str, Any]]]:
        """Query Pinecone for semantically similar documents."""
        if self.use_mock:
            return self._mock_search(query_vectors, top_k)

        results = []
        for qv in query_vectors:
            response = self.index.query(
                vector=qv.tolist(),
                top_k=top_k,
                include_metadata=True,
            )
            query_results = []
            for rank, match in enumerate(response.matches):
                query_results.append({
                    "rank": rank + 1,
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {},
                })
            results.append(query_results)
        return results

    def _mock_search(
        self, query_vectors: np.ndarray, top_k: int
    ) -> list[list[dict[str, Any]]]:
        if not self._mock_vectors:
            return [[] for _ in query_vectors]

        all_ids = list(self._mock_vectors.keys())
        all_vecs = np.array([self._mock_vectors[v][0] for v in all_ids])
        all_meta = [self._mock_vectors[v][1] for v in all_ids]

        results = []
        for qv in query_vectors:
            qv_norm = qv / (np.linalg.norm(qv) + 1e-8)
            vecs_norm = all_vecs / (np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-8)
            sims = vecs_norm @ qv_norm
            top_indices = np.argsort(sims)[::-1][:top_k]
            query_results = []
            for rank, idx in enumerate(top_indices):
                query_results.append({
                    "rank": rank + 1,
                    "id": all_ids[idx],
                    "score": float(sims[idx]),
                    "metadata": all_meta[idx],
                })
            results.append(query_results)
        return results

    def delete_all(self) -> None:
        if self.use_mock:
            self._mock_vectors.clear()
        elif self.index:
            self.index.delete(delete_all=True)

    def get_stats(self) -> dict[str, Any]:
        if self.use_mock:
            return {
                "index_name": self.index_name,
                "vector_count": len(self._mock_vectors),
                "mode": "mock",
            }
        stats = self.index.describe_index_stats()
        return {
            "index_name": self.index_name,
            "vector_count": stats.total_vector_count,
            "mode": "production",
        }
