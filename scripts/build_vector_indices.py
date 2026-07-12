"""Build FAISS, Milvus, and Pinecone vector indices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_search import DistributedDocumentSearchEngine
from src.vector_search.embeddings import DocumentEmbedder
from src.utils.config import ensure_directories, load_config


def main():
    config = load_config()
    ensure_directories(config)

    docs_path = Path(config["paths"]["data_sample"]) / "documents.json"
    if not docs_path.exists():
        print("Sample documents not found. Run generate_sample_data.py first.")
        sys.exit(1)

    with open(docs_path, encoding="utf-8") as f:
        documents = json.load(f)

    texts = [d["text"] for d in documents]
    metadata = [
        {"document_id": d["document_id"], "title": d["title"], "category": d["category"]}
        for d in documents
    ]
    ids = [d["document_id"] for d in documents]

    print(f"Embedding {len(texts)} documents...")
    embedder = DocumentEmbedder(dim=config["vector_search"]["embedding_dim"])
    vectors = embedder.embed_documents(texts)
    print(f"  Embedding shape: {vectors.shape}")

    engine = DistributedDocumentSearchEngine(config)
    counts = engine.build_all_indices(vectors, metadata, ids)

    faiss_dir = Path(config["paths"]["vector_indices"]) / "faiss"
    engine.save_faiss(faiss_dir)

    print("\n=== Semantic Search Demo ===")
    query = "compliance audit unusual transaction pattern"
    query_vec = embedder.embed_query(query)
    results = engine.semantic_search(query_vec.reshape(1, -1), top_k=3)

    for backend, hits in results.items():
        print(f"\n  [{backend.upper()}] Top results for: '{query}'")
        for hit in hits[0][:3]:
            if backend == "faiss":
                meta = hit.get("metadata", {})
                print(f"    #{hit['rank']} {meta.get('title', 'N/A')} (sim={hit['similarity']:.4f})")
            elif backend == "milvus":
                print(f"    #{hit['rank']} {hit.get('title', 'N/A')} (dist={hit.get('distance', 0):.4f})")
            else:
                meta = hit.get("metadata", {})
                print(f"    #{hit['rank']} {meta.get('title', 'N/A')} (score={hit.get('score', 0):.4f})")

    stats = engine.get_cluster_stats()
    summary = {"vector_counts": counts, "cluster_stats": stats, "demo_query": query}
    summary_path = Path(config["paths"]["vector_indices"]) / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nVector index summary saved to {summary_path}")
    print("Vector index build complete.")


if __name__ == "__main__":
    main()
