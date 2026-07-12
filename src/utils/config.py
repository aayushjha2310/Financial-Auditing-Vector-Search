"""Configuration loader for the sovereign financial auditing engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and merge environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, subpaths in config.get("paths", {}).items():
        config["paths"][key] = str(PROJECT_ROOT / subpaths)

    config["pinecone_api_key"] = os.getenv("PINECONE_API_KEY", "")
    config["milvus_host"] = os.getenv("MILVUS_HOST", config["vector_search"]["milvus"]["host"])
    config["milvus_port"] = int(os.getenv("MILVUS_PORT", config["vector_search"]["milvus"]["port"]))
    return config


def ensure_directories(config: dict[str, Any]) -> None:
    """Create required project directories."""
    for path_key in ("data_raw", "data_processed", "data_sample", "models", "logs", "vector_indices", "output"):
        Path(config["paths"][path_key]).mkdir(parents=True, exist_ok=True)


def get_epochs(config: dict[str, Any], pipeline_mode: bool = False) -> int:
    """Return training epochs for full or fast pipeline runs."""
    training = config.get("training", {})
    if pipeline_mode:
        return int(training.get("pipeline_epochs", training.get("epochs", 3)))
    return int(training.get("epochs", 10))
