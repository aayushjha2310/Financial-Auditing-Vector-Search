"""Multi-currency ledger processing with Pandas and NumPy."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


class LedgerProcessor:
    """Process complex multi-currency, multi-indexed financial ledger history."""

    COLUMNS = [
        "timestamp", "account_id", "currency", "debit", "credit",
        "balance", "fx_rate", "counterparty", "transaction_type", "risk_score",
    ]

    def __init__(self, currencies: list[str] | None = None):
        self.currencies = currencies or ["USD", "EUR", "GBP", "JPY", "CHF"]

    def load_ledger(self, path: str | Path) -> pd.DataFrame:
        """Load ledger CSV and parse multi-index structure."""
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["currency"] = df["currency"].astype("category")
        df["transaction_type"] = df["transaction_type"].astype("category")
        return df

    def create_multi_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build multi-index on (currency, account_id, timestamp)."""
        indexed = df.set_index(["currency", "account_id", "timestamp"]).sort_index()
        return indexed

    def normalize_to_base_currency(
        self, df: pd.DataFrame, base: str = "USD"
    ) -> pd.DataFrame:
        """Convert all amounts to base currency using FX rates."""
        result = df.copy()
        result["debit_base"] = result["debit"] * result["fx_rate"]
        result["credit_base"] = result["credit"] * result["fx_rate"]
        result["balance_base"] = result["balance"] * result["fx_rate"]
        result["net_flow"] = result["credit_base"] - result["debit_base"]
        return result

    def compute_rolling_features(
        self, df: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """Compute rolling volatility, momentum, and anomaly indicators."""
        grouped = df.groupby(["currency", "account_id"], observed=True)
        result = df.copy()
        result["rolling_volatility"] = grouped["net_flow"].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        result["rolling_mean"] = grouped["net_flow"].transform(
            lambda x: x.rolling(window, min_periods=1).mean().fillna(0)
        )
        result["z_score"] = (
            (result["net_flow"] - result["rolling_mean"])
            / (result["rolling_volatility"] + 1e-8)
        )
        return result

    def pivot_currency_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Pivot multi-currency flows into a 2D matrix for CNN input."""
        pivot = df.pivot_table(
            index="timestamp",
            columns="currency",
            values="net_flow",
            aggfunc="sum",
            fill_value=0.0,
            observed=True,
        )
        return pivot.values.astype(np.float32)

    def extract_sequence_features(
        self, df: pd.DataFrame, seq_len: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract LSTM/Transformer sequence features and risk labels."""
        feature_cols = [
            "debit_base", "credit_base", "balance_base", "net_flow",
            "rolling_volatility", "rolling_mean", "z_score", "fx_rate",
            "risk_score", "debit", "credit", "balance",
        ]
        available = [c for c in feature_cols if c in df.columns]
        values = df[available].values.astype(np.float32)
        labels = (df["z_score"].abs() > 2.0).astype(np.float32).values

        sequences, seq_labels = [], []
        for i in range(len(values) - seq_len):
            sequences.append(values[i : i + seq_len])
            seq_labels.append(labels[i + seq_len - 1])

        return np.array(sequences, dtype=np.float32), np.array(seq_labels, dtype=np.float32)

    def process_pipeline(self, path: str | Path, window: int = 20, seq_len: int = 60) -> dict:
        """Full data engineering pipeline."""
        df = self.load_ledger(path)
        df = self.normalize_to_base_currency(df)
        df = self.compute_rolling_features(df, window=window)
        currency_matrix = self.pivot_currency_matrix(df)
        sequences, labels = self.extract_sequence_features(df, seq_len=seq_len)
        return {
            "ledger": df,
            "multi_index": self.create_multi_index(df.reset_index()),
            "currency_matrix": currency_matrix,
            "sequences": sequences,
            "labels": labels,
        }
