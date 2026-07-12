"""Unified vector search interface across FAISS, Milvus, and Pinecone."""

from __future__ import annotations

from typing import Any

import numpy as np

from .faiss_engine import FAISSVectorEngine
from .milvus_engine import MilvusVectorEngine
from .pinecone_engine import PineconeVectorEngine


class DistributedDocumentSearchEngine:
    """
    Clustered vector index orchestrator combining FAISS (in-memory IVFPQ),
    Milvus (distributed HA), and Pinecone (managed production) backends.
    """

    def __init__(self, config: dict[str, Any]):
        vs_cfg = config["vector_search"]
        self.embedding_dim = vs_cfg["embedding_dim"]

        self.faiss = FAISSVectorEngine(
            embedding_dim=self.embedding_dim,
            **vs_cfg["faiss"],
        )
        self.milvus = MilvusVectorEngine(
            embedding_dim=self.embedding_dim,
            host=config.get("milvus_host", vs_cfg["milvus"]["host"]),
            port=config.get("milvus_port", vs_cfg["milvus"]["port"]),
            **{k: v for k, v in vs_cfg["milvus"].items() if k not in ("host", "port")},
        )
        self.pinecone = PineconeVectorEngine(
            embedding_dim=self.embedding_dim,
            api_key=config.get("pinecone_api_key", ""),
            **vs_cfg["pinecone"],
        )

    def build_all_indices(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Build vector indices across all three backends."""
        print(f"Building FAISS IVFPQ index ({len(vectors)} vectors)...")
        self.faiss.set_metadata(metadata)
        self.faiss.train_and_build(vectors)

        print("Building Milvus collection...")
        self.milvus.connect()
        milvus_count = self.milvus.insert(vectors, metadata)

        print("Building Pinecone index...")
        self.pinecone.initialize()
        pinecone_count = self.pinecone.upsert(vectors, metadata, ids)

        return {
            "faiss": self.faiss.vector_count,
            "milvus": milvus_count,
            "pinecone": pinecone_count,
        }

    def semantic_search(
        self, query_vectors: np.ndarray, top_k: int = 5
    ) -> dict[str, list]:
        """Execute parallel semantic verification across all backends."""
        return {
            "faiss": self.faiss.search(query_vectors, top_k),
            "milvus": self.milvus.search(query_vectors, top_k),
            "pinecone": self.pinecone.search(query_vectors, top_k),
        }

    def save_faiss(self, directory: str) -> None:
        self.faiss.save(directory)

    def load_faiss(self, directory: str) -> None:
        self.faiss.load(directory)

    def get_cluster_stats(self) -> dict[str, Any]:
        return {
            "faiss": {"vectors": self.faiss.vector_count, "type": "IVFPQ"},
            "milvus": self.milvus.get_stats(),
            "pinecone": self.pinecone.get_stats(),
        }
