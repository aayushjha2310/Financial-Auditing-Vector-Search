"""Convolutional Neural Network for 2D financial indicator grid analysis."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single convolution block with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class FinancialCNN(nn.Module):
    """
    CNN applied to 2D structural grid transformations of multi-variate
    financial indicators for anomaly/risk classification.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | None = None,
        num_classes: int = 1,
        grid_size: int = 8,
    ):
        super().__init__()
        channels = channels or [16, 32, 64]
        layers: list[nn.Module] = []
        prev = in_channels
        for ch in channels:
            layers.append(ConvBlock(prev, ch))
            layers.append(nn.MaxPool2d(2))
            prev = ch
        self.features = nn.Sequential(*layers)

        reduced = grid_size // (2 ** len(channels))
        flat_dim = channels[-1] * max(reduced, 1) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        return self.classifier(features), self.regressor(features)
