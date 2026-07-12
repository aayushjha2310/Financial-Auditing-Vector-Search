"""Feature engineering for financial indicator grids and document embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Transform ledger data into model-ready feature grids and vectors."""

    def __init__(self):
        self.scaler = StandardScaler()

    def build_indicator_grid(
        self, matrix: np.ndarray, height: int = 8, width: int = 8
    ) -> np.ndarray:
        """Reshape multi-variate financial indicators into 2D CNN grid."""
        n_samples = matrix.shape[0]
        target_size = height * width

        if matrix.shape[1] < target_size:
            padded = np.zeros((n_samples, target_size), dtype=np.float32)
            padded[:, : matrix.shape[1]] = matrix
            matrix = padded
        else:
            matrix = matrix[:, :target_size]

        grids = matrix.reshape(n_samples, 1, height, width)
        return grids.astype(np.float32)

    def create_sliding_windows(
        self, matrix: np.ndarray, window: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window grids with binary anomaly labels."""
        grids, labels = [], []
        for i in range(len(matrix) - window):
            window_data = matrix[i : i + window]
            grid = self.build_indicator_grid(
                window_data.reshape(1, -1),
                height=8,
                width=max(8, (window_data.size + 7) // 8),
            )
            if grid.shape[2] != 8 or grid.shape[3] != 8:
                flat = window_data.flatten()
                padded = np.zeros(64, dtype=np.float32)
                padded[: min(len(flat), 64)] = flat[:64]
                grid = padded.reshape(1, 1, 8, 8)

            volatility = np.std(window_data)
            label = 1.0 if volatility > np.percentile(
                [np.std(matrix[j : j + window]) for j in range(max(1, len(matrix) - window))], 90
            ) else 0.0
            grids.append(grid[0])
            labels.append(label)

        return np.array(grids, dtype=np.float32), np.array(labels, dtype=np.float32)

    def normalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Standard-scale sequence features across all timesteps."""
        n, t, f = sequences.shape
        flat = sequences.reshape(-1, f)
        scaled = self.scaler.fit_transform(flat)
        return scaled.reshape(n, t, f).astype(np.float32)

    def prepare_document_metadata(self, documents: list[dict]) -> pd.DataFrame:
        """Build searchable document metadata DataFrame."""
        return pd.DataFrame(documents)
