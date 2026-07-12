"""Train TensorFlow/Keras models for production comparison."""

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
from src.deep_learning.tensorflow import TensorFlowRiskModel
from src.utils.config import ensure_directories, get_epochs, load_config
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train TensorFlow/Keras financial models")
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

    sequences = fe.normalize_sequences(processed["sequences"])
    labels = processed["labels"]
    n = int(len(sequences) * config["training"]["train_split"])

    X_train, X_val = sequences[:n], sequences[n:]
    y_train, y_val = labels[:n], labels[n:]
    y_reg_train = np.abs(y_train) + np.random.randn(len(y_train)).astype(np.float32) * 0.01
    y_reg_val = np.abs(y_val) + np.random.randn(len(y_val)).astype(np.float32) * 0.01

    results = {}
    for model_type in ["dense", "lstm", "transformer"]:
        print(f"\n=== Training TensorFlow {model_type.upper()} Model ===")
        if model_type == "lstm":
            input_shape = (sequences.shape[1], sequences.shape[2])
            X_tr, X_v = X_train, X_val
        elif model_type == "transformer":
            input_shape = (sequences.shape[1], sequences.shape[2])
            X_tr, X_v = X_train, X_val
        else:
            input_shape = (sequences.shape[1] * sequences.shape[2],)
            X_tr = X_train.reshape(len(X_train), -1)
            X_v = X_val.reshape(len(X_val), -1)

        tf_model = TensorFlowRiskModel(
            input_shape=input_shape,
            model_type=model_type,
            dense_units=config["tensorflow"]["dense_units"],
            dropout=config["tensorflow"]["dropout"],
        )
        tf_model.train(
            X_tr, y_train, y_reg_train,
            X_v, y_val, y_reg_val,
            epochs=epochs,
            batch_size=config["training"]["batch_size"],
            log_dir=str(Path(config["paths"]["logs"]) / "tensorflow" / model_type),
        )
        eval_results = tf_model.evaluate(X_v, y_val, y_reg_val)
        tf_model.save(str(Path(config["paths"]["models"]) / f"tensorflow_{model_type}"))
        results[model_type] = eval_results
        print(f"  {model_type} accuracy: {eval_results['accuracy']:.4f}")

    summary_path = Path(config["paths"]["models"]) / "tensorflow_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTensorFlow training summary saved to {summary_path}")
    print("TensorFlow training complete.")


if __name__ == "__main__":
    main()
