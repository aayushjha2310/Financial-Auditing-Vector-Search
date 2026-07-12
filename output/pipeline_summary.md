# Sovereign Financial Auditing Engine — Pipeline Output

**Generated:** 2026-07-12 16:11:53

## Audit Report

| Metric | Value |
|--------|-------|
| Status | NORMAL |
| Entities Audited | 4940 |
| High Risk Entities | 0 (0.0%) |
| Average Risk Score | 0.0825 |

## PyTorch Models

| Model | Val Loss | Val Accuracy |
|-------|----------|--------------|
| CNN | 0.07239863770973898 | 0.9619619619619619 |
| LSTM | 0.026853630330503708 | 0.9858299595141701 |
| TRANSFORMER | 0.05174900881285148 | 0.9817813765182186 |

## Optimizer Comparison

- **ADAM**: final_loss=0.05462598545296538
- **SGD**: final_loss=0.31070112893658297
- **RMSPROP**: final_loss=0.029834541950314757

## TensorFlow/Keras Models

- **DENSE**: accuracy=0.9767206311225891, backend=keras_torch
- **LSTM**: accuracy=0.9827935099601746, backend=keras_torch
- **TRANSFORMER**: accuracy=0.9767206311225891, backend=keras_torch

## Vector Search Indices

- **FAISS:** 500 vectors
- **Milvus:** 500 vectors
- **Pinecone:** 500 vectors

## Output Artifacts

- `reports/` — JSON summaries and audit reports
- `charts/` — Training curves, audit charts, vector index bar chart
- `models/` — Saved model checkpoints
- `logs/` — TensorBoard and training logs
- `vector_indices/` — FAISS, Milvus, and Pinecone index artifacts