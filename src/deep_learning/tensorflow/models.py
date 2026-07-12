"""TensorFlow/Keras alternative models for production comparison.

Uses TensorFlow when available; falls back to Keras 3 with PyTorch backend
when MSVC runtime libraries are missing (common on Windows without admin).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_BACKEND = "unknown"
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    _ = tf.reduce_sum([1, 2, 3])
    _BACKEND = "tensorflow"
except (ImportError, AttributeError, ModuleNotFoundError):
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    from keras import layers

    _BACKEND = "keras_torch"


@dataclass
class TrainingHistory:
    """Minimal history object compatible with Keras History API."""

    history: dict[str, list[float]] = field(default_factory=dict)


class _StandaloneTensorBoard(keras.callbacks.Callback):
    """TensorBoard logging using raw protobuf events (no TensorFlow required)."""

    def __init__(self, log_dir: str):
        super().__init__()
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._writer = EventFileWriter(log_dir)

    def _make_scalar_event(self, tag: str, value: float, step: int):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata

        summary = Summary(
            value=[
                Summary.Value(
                    tag=tag,
                    simple_value=float(value),
                    metadata=SummaryMetadata(display_name=tag),
                )
            ]
        )
        return Event(step=step, wall_time=time.time(), summary=summary)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        for key, val in (logs or {}).items():
            if isinstance(val, (int, float)):
                event = self._make_scalar_event(key, val, epoch)
                self._writer.add_event(event)

    def on_train_end(self, logs=None):
        self._writer.flush()
        self._writer.close()


class TensorFlowRiskModel:
    """
    Production-grade Keras model for financial risk classification,
    used to compare against PyTorch custom architectures.
    """

    def __init__(
        self,
        input_shape: tuple,
        model_type: str = "dense",
        dense_units: list[int] | None = None,
        dropout: float = 0.3,
    ):
        self.model_type = model_type
        self.input_shape = input_shape
        self.dense_units = dense_units or [128, 64, 32]
        self.dropout = dropout
        self.backend = _BACKEND
        self.model = self._build_model()
        self.history: TrainingHistory | None = None

    def _build_model(self) -> keras.Model:
        inputs = keras.Input(shape=self.input_shape)

        if self.model_type == "cnn":
            x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Flatten()(x)
        elif self.model_type == "lstm":
            x = layers.LSTM(128, return_sequences=True)(inputs)
            x = layers.LSTM(64)(x)
        elif self.model_type == "transformer":
            x = layers.Dense(64)(inputs)
            attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.Flatten()(inputs)

        for units in self.dense_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.dropout)(x)

        cls_output = layers.Dense(1, activation="sigmoid", name="classification")(x)
        reg_output = layers.Dense(1, name="regression")(x)

        model = keras.Model(inputs=inputs, outputs=[cls_output, reg_output])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                "classification": "binary_crossentropy",
                "regression": "mse",
            },
            loss_weights={"classification": 0.6, "regression": 0.4},
            metrics={"classification": "accuracy"},
        )
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_cls_train: np.ndarray,
        y_reg_train: np.ndarray,
        X_val: np.ndarray,
        y_cls_val: np.ndarray,
        y_reg_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        log_dir: str = "logs/tensorflow",
    ) -> TrainingHistory:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        callbacks: list[keras.callbacks.Callback] = []
        if _BACKEND == "tensorflow":
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=False,
                )
            )
        else:
            callbacks.append(_StandaloneTensorBoard(log_dir))

        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        )

        print(f"  [Keras backend: {_BACKEND}]")
        hist = self.model.fit(
            X_train,
            {"classification": y_cls_train, "regression": y_reg_train},
            validation_data=(
                X_val,
                {"classification": y_cls_val, "regression": y_reg_val},
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        self.history = TrainingHistory(history=dict(hist.history))
        return self.history

    def evaluate(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray) -> dict:
        results = self.model.evaluate(
            X,
            {"classification": y_cls, "regression": y_reg},
            verbose=0,
        )
        if isinstance(results, dict):
            return {
                "loss": float(results.get("loss", 0)),
                "cls_loss": float(results.get("classification_loss", 0)),
                "reg_loss": float(results.get("regression_loss", 0)),
                "accuracy": float(results.get("classification_accuracy", 0)),
                "backend": self.backend,
            }
        return {
            "loss": results[0],
            "cls_loss": results[1],
            "reg_loss": results[2],
            "accuracy": results[3],
            "backend": self.backend,
        }

    def save(self, path: str) -> None:
        save_path = Path(path)
        if save_path.suffix not in (".keras", ".h5"):
            save_path = save_path.with_suffix(".keras")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        meta = {"backend": self.backend, "model_type": self.model_type}
        with open(save_path.with_suffix(".backend.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cls_pred, reg_pred = self.model.predict(X, verbose=0)
        return cls_pred, reg_pred


def get_backend() -> str:
    return _BACKEND
