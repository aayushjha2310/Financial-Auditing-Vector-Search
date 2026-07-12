"""Optimizer comparison utilities for Adam, SGD, and RMSprop."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import StepLR


@dataclass
class OptimizerMetrics:
    """Track optimizer convergence behavior."""

    name: str
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


def create_optimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    """Create Adam, SGD, or RMSprop optimizer."""
    params = model.parameters()
    optimizers = {
        "adam": lambda: Adam(params, lr=lr, weight_decay=weight_decay),
        "sgd": lambda: SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay),
        "rmsprop": lambda: RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay),
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers)}")
    return optimizers[name]()


class OptimizerComparator:
    """Compare convergence behavior of Adam, SGD, and RMSprop over financial loss landscapes."""

    def __init__(self, model_factory, lr: float = 0.001):
        self.model_factory = model_factory
        self.lr = lr
        self.results: dict[str, OptimizerMetrics] = {}

    def run_comparison(
        self,
        train_fn,
        data,
        epochs: int = 5,
        optimizers: list[str] | None = None,
    ) -> dict[str, OptimizerMetrics]:
        optimizers = optimizers or ["adam", "sgd", "rmsprop"]
        for opt_name in optimizers:
            model = self.model_factory()
            optimizer = create_optimizer(model, opt_name, lr=self.lr)
            scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
            metrics = OptimizerMetrics(name=opt_name)

            for epoch in range(epochs):
                loss, grad_norm = train_fn(model, optimizer, data)
                metrics.losses.append(loss)
                metrics.grad_norms.append(grad_norm)
                metrics.learning_rates.append(optimizer.param_groups[0]["lr"])
                scheduler.step()

            self.results[opt_name] = metrics
        return self.results

    def get_best_optimizer(self) -> str:
        if not self.results:
            return "adam"
        return min(self.results, key=lambda k: self.results[k].losses[-1])
