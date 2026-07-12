# Project 3: Sovereign Algorithmic High-Frequency Financial Auditing, Risk Analytics, & Distributed Document Search Engine

A highly secure deep learning analytics engine for ingesting multi-currency financial ledgers and corporate intelligence documents. Built for quantitative compliance and corporate treasury divisions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA ENGINEERING LAYER                           │
│  Pandas (multi-currency ledgers) │ NumPy (vector math)             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                 DEEP LEARNING ANALYTICS ENGINE                        │
│  PyTorch: CNN │ LSTM │ Transformer │ Backprop │ Multi-Task Loss      │
│  TensorFlow/Keras: Production comparison models                       │
│  TensorBoard: Weight/gradient visualization                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│              DISTRIBUTED VECTOR SEARCH CLUSTER                        │
│  FAISS (IVFPQ) │ Milvus (Lite/Server) │ Pinecone (Mock/Production)  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                 RISK ANALYTICS & AUDIT REPORTING                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack Coverage

| Category | Technologies |
|----------|-------------|
| Data Engineering | Pandas, NumPy |
| Deep Learning (PyTorch) | CNN, LSTM, Transformer, Backpropagation, Custom Losses, Adam/SGD/RMSprop |
| Deep Learning (TensorFlow) | Keras Dense/LSTM/Transformer models |
| Visualization | TensorBoard |
| Vector Search | FAISS (IVFPQ), Milvus, Pinecone |
| Embeddings | sentence-transformers |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full end-to-end pipeline
python scripts/run_full_pipeline.py
```

Or run individual steps:

```bash
python scripts/generate_sample_data.py      # Generate sample ledger + documents
python scripts/train_pytorch_models.py      # Train CNN, LSTM, Transformer
python scripts/train_tensorflow_models.py     # Train TF/Keras comparison models
python scripts/build_vector_indices.py        # Build FAISS, Milvus, Pinecone indices
```

## Project Structure

```
Project-5/
├── config/settings.yaml          # Central configuration
├── scripts/                      # Runnable pipeline scripts
├── src/
│   ├── data_engineering/         # Pandas/NumPy ledger processing
│   ├── deep_learning/
│   │   ├── pytorch/              # CNN, LSTM, Transformer, backprop, losses
│   │   └── tensorflow/           # Keras production models
│   ├── vector_search/            # FAISS, Milvus, Pinecone engines
│   └── risk_analytics/           # Risk scoring & audit reporting
├── data/sample/                  # Generated sample data
├── models/                       # Trained model checkpoints
├── logs/                         # TensorBoard logs
└── vector_indices/               # Persisted FAISS/Milvus indices
```

## Configuration

Edit `config/settings.yaml` to adjust training parameters, vector index settings, and risk thresholds.

For production Pinecone, copy `.env.example` to `.env` and set your API key:

```
PINECONE_API_KEY=your-key-here
```

Set `vector_search.pinecone.use_mock: false` in config to use real Pinecone.

### TensorFlow on Windows

Native TensorFlow requires the [Microsoft VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). If it is not installed, the project automatically uses **Keras 3 with PyTorch backend** for production comparison models — same Keras API, TensorBoard logging, and `.keras` model checkpoints.

## TensorBoard

```bash
tensorboard --logdir logs/
```

## Vector Search Backends

- **FAISS IVFPQ**: In-memory inverted file with product quantization for batch similarity sweeps
- **Milvus Lite**: Local embedded Milvus (no Docker required); switch `use_lite: false` for server mode
- **Pinecone**: Mock mode by default; set API key for production deployment

## Output

After running the pipeline:

- `models/` — Trained PyTorch and TensorFlow model checkpoints
- `logs/` — TensorBoard event files for weight/gradient monitoring
- `vector_indices/` — FAISS index + Milvus Lite database
- `data/processed/audit_report.json` — Compliance audit summary
