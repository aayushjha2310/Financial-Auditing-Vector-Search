"""LSTM Recurrent Neural Network for sequential financial anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class FinancialLSTM(nn.Module):
    """
    Multi-layer LSTM for tracking sequential ledger history and discovering
    structural multi-turn chronological anomalies.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            cls_logits, reg_output, attention_weights (last hidden state norms)
        """
        x = torch.relu(self.input_proj(x))
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            context = torch.cat([h_forward, h_backward], dim=-1)
        else:
            context = h_n[-1]

        cls_logits = self.classifier(context)
        reg_output = self.regressor(context)
        attn_weights = torch.norm(lstm_out, dim=-1)
        return cls_logits, reg_output, attn_weights
