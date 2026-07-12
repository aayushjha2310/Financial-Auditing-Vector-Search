"""Manual gradient tracking and backpropagation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class GradientSnapshot:
    """Snapshot of layer gradients at a training step."""

    step: int
    layer_name: str
    grad_mean: float
    grad_std: float
    grad_norm: float
    weight_mean: float
    weight_std: float


class ManualBackpropTracker:
    """
    Manual manipulation and tracking of layer gradients during backpropagation.
    Hooks into each layer to capture gradient distributions.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.snapshots: list[GradientSnapshot] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._step = 0

    def register_hooks(self) -> None:
        """Register backward hooks on all trainable parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._make_hook(name, param))
                self._hooks.append(hook)

    def _make_hook(self, name: str, param: nn.Parameter) -> Callable:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is not None:
                self.snapshots.append(
                    GradientSnapshot(
                        step=self._step,
                        layer_name=name,
                        grad_mean=grad.mean().item(),
                        grad_std=grad.std().item(),
                        grad_norm=grad.norm().item(),
                        weight_mean=param.data.mean().item(),
                        weight_std=param.data.std().item(),
                    )
                )
            return grad

        return hook

    def step(self) -> None:
        self._step += 1

    def compute_manual_gradients(
        self, loss: torch.Tensor, retain_graph: bool = False
    ) -> dict[str, torch.Tensor]:
        """Explicitly compute gradients via autograd and return them."""
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
        return grads

    def apply_gradient_clipping(self, max_norm: float = 1.0) -> float:
        """Clip gradients to prevent exploding gradients in RNN/Transformer."""
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm
        ).item()

    def get_gradient_summary(self) -> dict[str, dict[str, float]]:
        """Summarize latest gradient statistics per layer."""
        if not self.snapshots:
            return {}
        latest_step = max(s.step for s in self.snapshots)
        summary = {}
        for snap in self.snapshots:
            if snap.step == latest_step:
                summary[snap.layer_name] = {
                    "grad_mean": snap.grad_mean,
                    "grad_std": snap.grad_std,
                    "grad_norm": snap.grad_norm,
                }
        return summary

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def export_for_tensorboard(self) -> dict[str, float]:
        """Export gradient stats for TensorBoard histogram logging."""
        summary = self.get_gradient_summary()
        tb_data = {}
        for layer, stats in summary.items():
            safe_name = layer.replace(".", "/")
            tb_data[f"gradients/{safe_name}/mean"] = stats["grad_mean"]
            tb_data[f"gradients/{safe_name}/norm"] = stats["grad_norm"]
        return tb_data
