"""Collect all pipeline artifacts into the output/ folder."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config


def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _plot_training_loss(pytorch_summary: dict, charts_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    models = ["cnn", "lstm", "transformer"]
    titles = ["Financial CNN", "Financial LSTM", "Financial Transformer"]

    for ax, key, title in zip(axes, models, titles):
        if key in pytorch_summary and isinstance(pytorch_summary[key], dict):
            data = pytorch_summary[key]
            if "train_loss" in data and isinstance(data["train_loss"], list):
                ax.plot(data["train_loss"], label="Train Loss", marker="o")
                ax.plot(data["val_loss"], label="Val Loss", marker="s")
            else:
                ax.bar(["Train", "Val"], [data.get("train_loss", 0), data.get("val_loss", 0)])
            acc = data.get("val_accuracy", 0)
            ax.set_title(f"{title}\nVal Acc: {acc:.2%}" if isinstance(acc, float) else title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("PyTorch Model Training Loss", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = charts_dir / "training_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_optimizer_comparison(opt_data: dict, charts_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"adam": "#2196F3", "sgd": "#FF5722", "rmsprop": "#4CAF50"}
    for name, metrics in opt_data.items():
        losses = metrics.get("losses", [])
        if losses:
            ax.plot(
                range(1, len(losses) + 1),
                losses,
                marker="o",
                label=name.upper(),
                color=colors.get(name, "gray"),
                linewidth=2,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Optimizer Convergence Comparison (Adam vs SGD vs RMSprop)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = charts_dir / "optimizer_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_keras_accuracy(tf_summary: dict, charts_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(tf_summary.keys())
    accuracies = [tf_summary[m].get("accuracy", 0) for m in models]
    losses = [tf_summary[m].get("loss", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#1976D2")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, losses, width, label="Loss", color="#D32F2F", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")
    ax.set_title("TensorFlow/Keras Model Accuracy")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    path = charts_dir / "keras_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_currency_exposure(audit: dict, charts_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    exposure = audit.get("currency_exposure", {})
    if exposure:
        currencies = list(exposure.keys())
        values = [v / 1e9 for v in exposure.values()]
        bars = ax.bar(
            currencies,
            values,
            color=["#1565C0", "#2E7D32", "#F57C00", "#C62828", "#6A1B9A"],
        )
        ax.set_ylabel("Exposure (Billions USD)")
        ax.set_title("Multi-Currency Portfolio Exposure")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}B",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    path = charts_dir / "currency_exposure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_top_risks(audit: dict, charts_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    top_risks = audit.get("top_risks", [])[:10]
    if top_risks:
        entities = [r["entity"][:12] for r in top_risks]
        scores = [r["score"] for r in top_risks]
        ax.barh(entities[::-1], scores[::-1], color="#E53935", alpha=0.8)
        ax.set_xlabel("Risk Score")
        ax.set_title("Top 10 Risk Entities")
        ax.axvline(
            x=audit.get("average_risk_score", 0.5),
            color="blue",
            linestyle="--",
            label=f"Avg: {audit.get('average_risk_score', 0):.3f}",
        )
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = charts_dir / "top_risks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_risk_distribution(audit: dict, charts_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    top_risks = audit.get("top_risks", [])
    if top_risks:
        scores = [r["score"] for r in top_risks]
        ax.hist(scores, bins=10, color="#7B1FA2", alpha=0.75, edgecolor="white")
        ax.axvline(
            audit.get("average_risk_score", 0),
            color="blue",
            linestyle="--",
            label=f"Avg: {audit.get('average_risk_score', 0):.3f}",
        )
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Entity Count")
        ax.set_title("Risk Score Distribution (Top Entities)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = charts_dir / "risk_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_vector_index_counts(vector_summary: dict, charts_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = vector_summary.get("vector_counts", {})
    if counts:
        engines = list(counts.keys())
        values = list(counts.values())
        ax.bar([e.upper() for e in engines], values, color=["#00897B", "#5E35B1", "#F9A825"])
        ax.set_ylabel("Vector Count")
        ax.set_title("Vector Search Index Counts")
        for i, val in enumerate(values):
            ax.text(i, val, str(val), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = charts_dir / "vector_index_counts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_audit_report_txt(audit: dict, path: Path) -> None:
    lines = [
        "=" * 60,
        "  SOVEREIGN FINANCIAL AUDITING ENGINE — AUDIT REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"  Status:              {audit.get('status', 'N/A')}",
        f"  Entities Audited:    {audit.get('total_entities_audited', 0)}",
        f"  High Risk Entities:  {audit.get('high_risk_entities', 0)} ({audit.get('high_risk_percentage', 0)}%)",
        f"  Average Risk Score:  {audit.get('average_risk_score', 0)}",
        f"  Max Volatility:      {audit.get('max_volatility', 'N/A')}",
        "",
        "CURRENCY EXPOSURE",
    ]
    for currency, amount in (audit.get("currency_exposure") or {}).items():
        lines.append(f"  {currency}: {amount:,.2f}")

    lines.extend(["", "TOP RISKS"])
    for risk in audit.get("top_risks", [])[:10]:
        lines.append(f"  {risk.get('entity', 'N/A'):20s}  score={risk.get('score', 0):.4f}  {risk.get('recommendation', '')}")

    lines.append("=" * 60)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_readme(output_dir: Path, manifest: dict) -> None:
    lines = [
        "SOVEREIGN FINANCIAL AUDITING & RISK ANALYTICS ENGINE",
        "Pipeline Output Directory",
        "=" * 60,
        "",
        "This folder contains all artifacts produced by the end-to-end pipeline.",
        "",
        "FILES:",
        "  pipeline_log.txt              Full pipeline execution log",
        "  audit_report.json             Machine-readable audit report",
        "  audit_report.txt              Human-readable audit summary",
        "  pytorch_training_summary.json PyTorch model training metrics",
        "  tensorflow_training_summary.json  Keras model training metrics",
        "  vector_build_summary.json     Vector index build summary",
        "  sample_ledger_history.csv     Sample ledger data",
        "  sample_documents.json         Sample financial documents",
        "  pipeline_summary.json         Manifest of all collected artifacts",
        "",
        "CHARTS/:",
        "  training_loss.png             PyTorch training/validation loss curves",
        "  optimizer_comparison.png      Adam vs SGD vs RMSprop convergence",
        "  keras_accuracy.png            TensorFlow/Keras model accuracy",
        "  currency_exposure.png         Multi-currency portfolio exposure",
        "  top_risks.png                 Top 10 highest-risk entities",
        "  risk_distribution.png         Risk score histogram",
        "  vector_index_counts.png       FAISS/Milvus/Pinecone vector counts",
        "",
        f"Generated: {manifest.get('generated_at', 'N/A')}",
        f"Audit Status: {manifest.get('audit_status', 'N/A')}",
        "=" * 60,
    ]
    (output_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def collect_all_outputs(config: dict[str, Any] | None = None, pipeline_log: str = "") -> Path:
    """Gather all pipeline artifacts into the output/ directory."""
    config = config or load_config()
    output_dir = Path(config["paths"]["output"])
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(config["paths"]["models"])
    data_processed = Path(config["paths"]["data_processed"])
    data_sample = Path(config["paths"]["data_sample"])
    vector_indices = Path(config["paths"]["vector_indices"])

    pytorch_summary = _load_json(models_dir / "pytorch_training_summary.json") or {}
    tf_summary = _load_json(models_dir / "tensorflow_training_summary.json") or {}
    audit = _load_json(data_processed / "audit_report.json") or {}
    vector_summary = _load_json(vector_indices / "build_summary.json") or {}

    if pipeline_log:
        (output_dir / "pipeline_log.txt").write_text(pipeline_log, encoding="utf-8")

    with open(output_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    _write_audit_report_txt(audit, output_dir / "audit_report.txt")

    for src_name, dst_name in (
        ("pytorch_training_summary.json", "pytorch_training_summary.json"),
        ("tensorflow_training_summary.json", "tensorflow_training_summary.json"),
    ):
        src = models_dir / src_name
        if src.exists():
            shutil.copy2(src, output_dir / dst_name)

    vector_src = vector_indices / "build_summary.json"
    if vector_src.exists():
        shutil.copy2(vector_src, output_dir / "vector_build_summary.json")

    for src_name, dst_name in (
        ("ledger_history.csv", "sample_ledger_history.csv"),
        ("documents.json", "sample_documents.json"),
    ):
        src = data_sample / src_name
        if src.exists():
            shutil.copy2(src, output_dir / dst_name)

    chart_paths: list[str] = []
    if pytorch_summary:
        chart_paths.append(str(_plot_training_loss(pytorch_summary, charts_dir).relative_to(output_dir)))
        opt_data = pytorch_summary.get("optimizer_comparison")
        if opt_data:
            chart_paths.append(str(_plot_optimizer_comparison(opt_data, charts_dir).relative_to(output_dir)))
    if tf_summary:
        chart_paths.append(str(_plot_keras_accuracy(tf_summary, charts_dir).relative_to(output_dir)))
    if audit:
        chart_paths.append(str(_plot_currency_exposure(audit, charts_dir).relative_to(output_dir)))
        chart_paths.append(str(_plot_top_risks(audit, charts_dir).relative_to(output_dir)))
        chart_paths.append(str(_plot_risk_distribution(audit, charts_dir).relative_to(output_dir)))
    if vector_summary:
        chart_paths.append(str(_plot_vector_index_counts(vector_summary, charts_dir).relative_to(output_dir)))

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "project": config.get("project", {}),
        "audit_status": audit.get("status"),
        "entities_audited": audit.get("total_entities_audited"),
        "high_risk_entities": audit.get("high_risk_entities"),
        "pytorch_models": {k: v for k, v in pytorch_summary.items() if k != "optimizer_comparison"},
        "tensorflow_models": tf_summary,
        "vector_indices": vector_summary,
        "charts": chart_paths,
        "files": [
            "pipeline_log.txt",
            "audit_report.json",
            "audit_report.txt",
            "pytorch_training_summary.json",
            "tensorflow_training_summary.json",
            "vector_build_summary.json",
            "sample_ledger_history.csv",
            "sample_documents.json",
            "README.txt",
        ] + chart_paths,
    }

    with open(output_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_readme(output_dir, manifest)

    print(f"\nAll outputs saved to: {output_dir}")
    return output_dir
