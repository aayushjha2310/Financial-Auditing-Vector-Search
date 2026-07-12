"""PyTorch training framework with TensorBoard integration."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .backprop import ManualBackpropTracker
from .losses import MultiTaskFinancialLoss
from .optimizers import create_optimizer


class PyTorchTrainer:
    """Unified trainer for CNN, LSTM, and Transformer models."""

    def __init__(
        self,
        model: nn.Module,
        device: str | None = None,
        log_dir: str | Path = "logs/pytorch",
        optimizer_name: str = "adam",
        lr: float = 0.001,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = create_optimizer(model, optimizer_name, lr=lr)
        self.criterion = MultiTaskFinancialLoss()
        self.tracker = ManualBackpropTracker(model)
        self.tracker.register_hooks()
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.global_step = 0

    def _make_loader(
        self,
        X: np.ndarray,
        y_cls: np.ndarray,
        y_reg: np.ndarray | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        if y_reg is None:
            y_reg = y_cls.copy()
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_cls, dtype=torch.float32),
            torch.tensor(y_reg, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "cls_loss": 0.0, "reg_loss": 0.0}
        n_batches = 0

        for batch_x, batch_cls, batch_reg in loader:
            batch_x = batch_x.to(self.device)
            batch_cls = batch_cls.to(self.device)
            batch_reg = batch_reg.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)

            if isinstance(outputs, tuple):
                cls_logits = outputs[0]
                reg_preds = outputs[1]
            else:
                cls_logits = outputs
                reg_preds = outputs

            loss, loss_dict = self.criterion(cls_logits, batch_cls, reg_preds, batch_reg)
            self.tracker.compute_manual_gradients(loss)
            grad_norm = self.tracker.apply_gradient_clipping(max_norm=1.0)
            self.optimizer.step()
            self.tracker.step()

            for k in totals:
                key = k if k != "loss" else "total_loss"
                totals[k] += loss_dict.get(key, loss_dict.get("total_loss", 0))
            n_batches += 1

        # Log gradient stats once per epoch (not every batch) for speed
        tb_grads = self.tracker.export_for_tensorboard()
        for tag, val in tb_grads.items():
            self.writer.add_scalar(tag, val, self.global_step)
        self.global_step += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals = {"loss": 0.0, "cls_loss": 0.0, "reg_loss": 0.0}
        correct, total = 0, 0
        n_batches = 0

        for batch_x, batch_cls, batch_reg in loader:
            batch_x = batch_x.to(self.device)
            batch_cls = batch_cls.to(self.device)
            batch_reg = batch_reg.to(self.device)

            outputs = self.model(batch_x)
            cls_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            reg_preds = outputs[1] if isinstance(outputs, tuple) else outputs

            loss, loss_dict = self.criterion(cls_logits, batch_cls, reg_preds, batch_reg)
            totals["loss"] += loss_dict["total_loss"]
            totals["cls_loss"] += loss_dict["cls_loss"]
            totals["reg_loss"] += loss_dict["reg_loss"]
            n_batches += 1

            preds = (torch.sigmoid(cls_logits.squeeze()) > 0.5).float()
            correct += (preds == batch_cls).sum().item()
            total += batch_cls.size(0)

        metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
        metrics["accuracy"] = correct / max(total, 1)
        return metrics

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> dict[str, list[float]]:
        y_reg_train = np.abs(y_train) + np.random.randn(len(y_train)).astype(np.float32) * 0.01
        y_reg_val = np.abs(y_val) + np.random.randn(len(y_val)).astype(np.float32) * 0.01

        train_loader = self._make_loader(X_train, y_train, y_reg_train, batch_size)
        val_loader = self._make_loader(X_val, y_val, y_reg_val, batch_size, shuffle=False)

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_accuracy": [],
        }

        for epoch in tqdm(range(epochs), desc="Training"):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            self.writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], epoch)

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f"weights/{name}", param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)

        return history

    def save_model(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def close(self) -> None:
        self.tracker.remove_hooks()
        self.writer.close()
