"""Train PyTorch CNN, LSTM, and Transformer models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engineering import FeatureEngineer, LedgerProcessor
from src.deep_learning.pytorch import (
    FinancialCNN,
    FinancialLSTM,
    FinancialTransformer,
    OptimizerComparator,
    PyTorchTrainer,
)
from src.utils.config import ensure_directories, get_epochs, load_config
from src.utils import set_seed


def split_data(X, y, train_split=0.8):
    n = int(len(X) * train_split)
    return X[:n], X[n:], y[:n], y[n:]


def train_cnn(config, grids, labels, epochs: int):
    print("\n=== Training Financial CNN ===")
    X_train, X_val, y_train, y_val = split_data(grids, labels, config["training"]["train_split"])
    model = FinancialCNN(channels=config["pytorch"]["cnn_channels"])
    trainer = PyTorchTrainer(
        model,
        log_dir=Path(config["paths"]["logs"]) / "pytorch" / "cnn",
        optimizer_name="adam",
        lr=config["training"]["learning_rate"],
    )
    history = trainer.fit(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=config["training"]["batch_size"],
    )
    trainer.save_model(Path(config["paths"]["models"]) / "pytorch_cnn.pt")
    trainer.close()
    return history


def train_lstm(config, sequences, labels, epochs: int):
    print("\n=== Training Financial LSTM ===")
    fe = FeatureEngineer()
    sequences = fe.normalize_sequences(sequences)
    X_train, X_val, y_train, y_val = split_data(sequences, labels, config["training"]["train_split"])
    model = FinancialLSTM(
        input_dim=sequences.shape[-1],
        hidden_dim=config["pytorch"]["lstm_hidden"],
        num_layers=config["pytorch"]["lstm_layers"],
    )
    trainer = PyTorchTrainer(
        model,
        log_dir=Path(config["paths"]["logs"]) / "pytorch" / "lstm",
        optimizer_name="adam",
        lr=config["training"]["learning_rate"],
    )
    history = trainer.fit(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=config["training"]["batch_size"],
    )
    trainer.save_model(Path(config["paths"]["models"]) / "pytorch_lstm.pt")
    trainer.close()
    return history


def train_transformer(config, sequences, labels, epochs: int):
    print("\n=== Training Financial Transformer ===")
    fe = FeatureEngineer()
    sequences = fe.normalize_sequences(sequences)
    X_train, X_val, y_train, y_val = split_data(sequences, labels, config["training"]["train_split"])
    model = FinancialTransformer(
        input_dim=sequences.shape[-1],
        d_model=config["pytorch"]["transformer_d_model"],
        nhead=config["pytorch"]["transformer_nhead"],
        num_layers=config["pytorch"]["transformer_layers"],
    )
    trainer = PyTorchTrainer(
        model,
        log_dir=Path(config["paths"]["logs"]) / "pytorch" / "transformer",
        optimizer_name="adam",
        lr=config["training"]["learning_rate"],
    )
    history = trainer.fit(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=config["training"]["batch_size"],
    )
    trainer.save_model(Path(config["paths"]["models"]) / "pytorch_transformer.pt")
    trainer.close()
    return history


def compare_optimizers(config, sequences, labels, epochs: int):
    print("\n=== Optimizer Comparison (Adam vs SGD vs RMSprop) ===")
    fe = FeatureEngineer()
    sequences = fe.normalize_sequences(sequences)
    X_train, X_val, y_train, y_val = split_data(sequences, labels, config["training"]["train_split"])

    def train_fn(model, optimizer, data):
        trainer = PyTorchTrainer(model, optimizer_name="adam")
        trainer.optimizer = optimizer
        loader = trainer._make_loader(X_train, y_train, batch_size=64)
        metrics = trainer.train_epoch(loader)
        grad_norm = trainer.tracker.apply_gradient_clipping()
        trainer.close()
        return metrics["loss"], grad_norm

    comparator = OptimizerComparator(
        lambda: FinancialLSTM(input_dim=sequences.shape[-1], hidden_dim=64, num_layers=1),
        lr=config["training"]["learning_rate"],
    )
    results = comparator.run_comparison(train_fn, None, epochs=epochs)
    summary = {
        name: {"final_loss": m.losses[-1], "losses": m.losses}
        for name, m in results.items()
    }
    print(f"  Best optimizer: {comparator.get_best_optimizer()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch financial models")
    parser.add_argument("--pipeline", action="store_true", help="Use pipeline_epochs for faster training")
    args = parser.parse_args()
    pipeline_mode = args.pipeline or os.getenv("PIPELINE_MODE", "").lower() in ("1", "true", "yes")

    config = load_config()
    ensure_directories(config)
    set_seed(config["training"]["random_seed"])
    epochs = get_epochs(config, pipeline_mode=pipeline_mode)
    print(f"Training with {epochs} epoch(s)" + (" (pipeline mode)" if pipeline_mode else ""))

    ledger_path = Path(config["paths"]["data_sample"]) / "ledger_history.csv"
    if not ledger_path.exists():
        print("Sample data not found. Run generate_sample_data.py first.")
        sys.exit(1)

    processor = LedgerProcessor(currencies=config["data"]["currencies"])
    fe = FeatureEngineer()
    processed = processor.process_pipeline(
        ledger_path,
        window=config["risk"]["volatility_window"],
        seq_len=config["data"]["sequence_length"],
    )

    grids, grid_labels = fe.create_sliding_windows(processed["currency_matrix"])
    sequences = processed["sequences"]
    seq_labels = processed["labels"]

    results = {}
    results["cnn"] = train_cnn(config, grids, grid_labels, epochs)
    results["lstm"] = train_lstm(config, sequences, seq_labels, epochs)
    results["transformer"] = train_transformer(config, sequences, seq_labels, epochs)
    results["optimizer_comparison"] = compare_optimizers(config, sequences, seq_labels, epochs)

    summary_path = Path(config["paths"]["models"]) / "pytorch_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            k: {mk: (mv[-1] if isinstance(mv, list) else mv) for mk, mv in v.items()}
            if isinstance(v, dict) and "train_loss" in v
            else v
            for k, v in results.items()
        }, f, indent=2, default=str)
    print(f"\nTraining summary saved to {summary_path}")
    print("PyTorch training complete.")


if __name__ == "__main__":
    main()
