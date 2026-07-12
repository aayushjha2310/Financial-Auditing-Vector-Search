"""Custom multi-task loss functions for financial risk prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskFinancialLoss(nn.Module):
    """
    Combined cross-entropy (classification) and MSE (regression) loss
    for multi-task financial risk mitigation.
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_preds: torch.Tensor,
        reg_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls_loss = self.ce(cls_logits.squeeze(-1), cls_targets)
        reg_loss = self.mse(reg_preds.squeeze(-1), reg_targets)
        total = self.alpha * cls_loss + self.beta * reg_loss
        return total, {
            "cls_loss": cls_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": total.item(),
        }


class FocalLoss(nn.Module):
    """Focal loss for imbalanced anomaly detection."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()
