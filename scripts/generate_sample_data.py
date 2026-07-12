"""Generate sample financial ledger and document data."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ensure_directories, load_config


DOCUMENT_TEMPLATES = [
    "Quarterly treasury report for {currency} exposure showing {amount} million in net position.",
    "Compliance audit finding: unusual transaction pattern detected in account {account} during {period}.",
    "Risk assessment memo: FX hedging strategy review for {currency} portfolio segment {segment}.",
    "Internal control evaluation: segregation of duties verification for treasury operations unit {unit}.",
    "Regulatory filing: Basel III capital adequacy ratio analysis for Q{quarter} {year}.",
    "Counterparty credit review: exposure limit breach warning for entity {entity} rated {rating}.",
    "Multi-currency reconciliation report: variance of {variance}% identified in {currency} ledger.",
    "Anti-money laundering alert: suspicious activity report filed for transaction series {series}.",
]


def generate_ledger(n_rows: int, currencies: list[str], seed: int = 42) -> pd.DataFrame:
    """Generate multi-currency multi-indexed ledger history."""
    rng = np.random.RandomState(seed)
    base_date = datetime(2023, 1, 1)
    accounts = [f"ACC-{i:04d}" for i in range(1, 51)]
    tx_types = ["wire", "fx_swap", "deposit", "withdrawal", "interest", "fee"]
    counterparties = [f"CP-{i:03d}" for i in range(1, 21)]

    fx_rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "CHF": 1.12}

    rows = []
    for i in range(n_rows):
        currency = rng.choice(currencies)
        debit = round(abs(rng.randn()) * 50000, 2) if rng.rand() > 0.5 else 0.0
        credit = round(abs(rng.randn()) * 50000, 2) if debit == 0 else 0.0
        balance = round(rng.uniform(100000, 5000000), 2)
        risk = round(min(abs(rng.randn()) * 0.3, 1.0), 4)

        rows.append({
            "timestamp": base_date + timedelta(hours=i * 4),
            "account_id": rng.choice(accounts),
            "currency": currency,
            "debit": debit,
            "credit": credit,
            "balance": balance,
            "fx_rate": fx_rates[currency],
            "counterparty": rng.choice(counterparties),
            "transaction_type": rng.choice(tx_types),
            "risk_score": risk,
        })

    return pd.DataFrame(rows)


def generate_documents(n_docs: int, seed: int = 42) -> list[dict]:
    """Generate corporate financial intelligence documents."""
    rng = np.random.RandomState(seed)
    currencies = ["USD", "EUR", "GBP", "JPY", "CHF"]
    categories = ["compliance", "treasury", "risk", "audit", "regulatory"]

    documents = []
    for i in range(n_docs):
        template = DOCUMENT_TEMPLATES[i % len(DOCUMENT_TEMPLATES)]
        text = template.format(
            currency=rng.choice(currencies),
            amount=round(rng.uniform(1, 500), 1),
            account=f"ACC-{rng.randint(1, 50):04d}",
            period=f"Q{rng.randint(1, 4)} 2024",
            segment=rng.randint(1, 10),
            unit=rng.randint(1, 5),
            quarter=rng.randint(1, 4),
            year=2024,
            entity=f"CP-{rng.randint(1, 20):03d}",
            rating=rng.choice(["AAA", "AA", "A", "BBB", "BB"]),
            variance=round(rng.uniform(0.1, 5.0), 2),
            series=f"TXN-{rng.randint(1000, 9999)}",
        )
        documents.append({
            "document_id": f"DOC-{i:05d}",
            "title": f"Financial Document {i + 1}",
            "category": categories[i % len(categories)],
            "text": text,
            "created_at": (datetime(2024, 1, 1) + timedelta(days=i)).isoformat(),
        })
    return documents


def main():
    config = load_config()
    ensure_directories(config)
    seed = config["training"]["random_seed"]
    n_rows = config["data"]["ledger_rows"]
    n_docs = config["data"]["document_count"]
    currencies = config["data"]["currencies"]

    print(f"Generating {n_rows} ledger rows...")
    ledger = generate_ledger(n_rows, currencies, seed)
    ledger_path = Path(config["paths"]["data_sample"]) / "ledger_history.csv"
    ledger.to_csv(ledger_path, index=False)
    print(f"  Saved: {ledger_path}")

    print(f"Generating {n_docs} corporate documents...")
    documents = generate_documents(n_docs, seed)
    docs_path = Path(config["paths"]["data_sample"]) / "documents.json"
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
    print(f"  Saved: {docs_path}")

    print("Sample data generation complete.")


if __name__ == "__main__":
    main()
