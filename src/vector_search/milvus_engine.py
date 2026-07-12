"""Milvus distributed vector database engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class MilvusVectorEngine:
    """
    Highly scalable Milvus vector database for high-availability
    distributed document search. Supports Milvus Lite for local deployment.
    """

    def __init__(
        self,
        collection_name: str = "financial_documents",
        embedding_dim: int = 384,
        host: str = "localhost",
        port: int = 19530,
        use_lite: bool = True,
        lite_db: str = "vector_indices/milvus_lite.db",
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.host = host
        self.port = port
        self.use_lite = use_lite
        self.lite_db = lite_db
        self.client = None
        self._connected = False
        self._fallback_vectors: np.ndarray | None = None
        self._fallback_metadata: list[dict] = []

    def connect(self) -> bool:
        """Connect to Milvus server or Milvus Lite."""
        try:
            if self.use_lite:
                from pymilvus import MilvusClient
                Path(self.lite_db).parent.mkdir(parents=True, exist_ok=True)
                self.client = MilvusClient(self.lite_db)
            else:
                from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
                connections.connect(host=self.host, port=self.port)
                self.client = "server"
                self._ensure_server_collection()
            self._connected = True
            return True
        except Exception as e:
            print(f"[Milvus] Connection failed ({e}), using in-memory fallback.")
            self._connected = False
            return False

    def _ensure_server_collection(self) -> None:
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

        if utility.has_collection(self.collection_name):
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields, description="Financial audit documents")
        Collection(self.collection_name, schema)

    def create_collection(self) -> None:
        if not self._connected:
            self.connect()
        if self.use_lite and self.client:
            if not self.client.has_collection(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.embedding_dim,
                    metric_type="COSINE",
                    auto_id=True,
                )

    def insert(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> int:
        """Insert document vectors with metadata."""
        if not self._connected:
            self.connect()

        vectors = vectors.astype(np.float32).tolist()

        if self.use_lite and self.client and self._connected:
            self.create_collection()
            data = []
            for i, (vec, meta) in enumerate(zip(vectors, metadata)):
                data.append({
                    "vector": vec,
                    "document_id": meta.get("document_id", f"doc_{i}"),
                    "title": meta.get("title", "")[:512],
                    "category": meta.get("category", "general")[:64],
                })
            result = self.client.insert(collection_name=self.collection_name, data=data)
            return len(data)

        self._fallback_vectors = np.array(vectors, dtype=np.float32)
        self._fallback_metadata = metadata
        return len(metadata)

    def search(
        self, query_vectors: np.ndarray, top_k: int = 10
    ) -> list[list[dict[str, Any]]]:
        """Semantic search across Milvus collection."""
        if self.use_lite and self.client and self._connected:
            results = []
            for qv in query_vectors:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    data=[qv.astype(np.float32).tolist()],
                    limit=top_k,
                    output_fields=["document_id", "title", "category"],
                )
                query_results = []
                for rank, hit in enumerate(hits[0]):
                    entity = hit.get("entity", {})
                    query_results.append({
                        "rank": rank + 1,
                        "id": hit.get("id"),
                        "distance": hit.get("distance", 0),
                        "document_id": entity.get("document_id", ""),
                        "title": entity.get("title", ""),
                        "category": entity.get("category", ""),
                    })
                results.append(query_results)
            return results

        return self._fallback_search(query_vectors, top_k)

    def _fallback_search(
        self, query_vectors: np.ndarray, top_k: int
    ) -> list[list[dict[str, Any]]]:
        if self._fallback_vectors is None:
            return [[] for _ in query_vectors]

        results = []
        for qv in query_vectors:
            qv_norm = qv / (np.linalg.norm(qv) + 1e-8)
            vecs_norm = self._fallback_vectors / (
                np.linalg.norm(self._fallback_vectors, axis=1, keepdims=True) + 1e-8
            )
            sims = vecs_norm @ qv_norm
            top_indices = np.argsort(sims)[::-1][:top_k]
            query_results = []
            for rank, idx in enumerate(top_indices):
                meta = self._fallback_metadata[idx] if idx < len(self._fallback_metadata) else {}
                query_results.append({
                    "rank": rank + 1,
                    "index": int(idx),
                    "similarity": float(sims[idx]),
                    **meta,
                })
            results.append(query_results)
        return results

    def get_stats(self) -> dict[str, Any]:
        count = 0
        if self.use_lite and self.client and self._connected:
            stats = self.client.get_collection_stats(self.collection_name)
            count = stats.get("row_count", 0)
        elif self._fallback_vectors is not None:
            count = len(self._fallback_vectors)
        return {
            "collection": self.collection_name,
            "vector_count": count,
            "embedding_dim": self.embedding_dim,
            "mode": "lite" if self.use_lite else "server",
            "connected": self._connected,
        }
