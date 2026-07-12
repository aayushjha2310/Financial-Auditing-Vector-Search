"""Collect pipeline outputs (text, JSON, visualizations) into output/ folder."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _plot_training_curves(pytorch_summary: dict, output_dir: Path) -> Path:
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

    fig.suptitle("PyTorch Model Training Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "charts" / "pytorch_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_optimizer_comparison(opt_data: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"adam": "#2196F3", "sgd": "#FF5722", "rmsprop": "#4CAF50"}
    for name, metrics in opt_data.items():
        losses = metrics.get("losses", [])
        if losses:
            ax.plot(range(1, len(losses) + 1), losses, marker="o", label=name.upper(),
                    color=colors.get(name, "gray"), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Optimizer Convergence Comparison (Adam vs SGD vs RMSprop)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / "charts" / "optimizer_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_audit_charts(audit: dict, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    exposure = audit.get("currency_exposure", {})
    if exposure:
        currencies = list(exposure.keys())
        values = [v / 1e9 for v in exposure.values()]
        bars = axes[0].bar(currencies, values, color=["#1565C0", "#2E7D32", "#F57C00", "#C62828", "#6A1B9A"])
        axes[0].set_ylabel("Exposure (Billions USD)")
        axes[0].set_title("Multi-Currency Portfolio Exposure")
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}B",
                         ha="center", va="bottom", fontsize=9)

    top_risks = audit.get("top_risks", [])[:10]
    if top_risks:
        entities = [r["entity"][:12] for r in top_risks]
        scores = [r["score"] for r in top_risks]
        axes[1].barh(entities[::-1], scores[::-1], color="#E53935", alpha=0.8)
        axes[1].set_xlabel("Risk Score")
        axes[1].set_title("Top 10 Risk Entities")
        axes[1].axvline(x=audit.get("average_risk_score", 0.5), color="blue", linestyle="--",
                        label=f"Avg: {audit.get('average_risk_score', 0):.3f}")
        axes[1].legend(fontsize=8)

    fig.suptitle(f"Audit & Risk Dashboard — Status: {audit.get('status', 'N/A')}", fontweight="bold")
    plt.tight_layout()
    path = output_dir / "charts" / "risk_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_vector_index_bar_chart(vector_summary: dict, output_dir: Path) -> Path:
    counts = vector_summary.get("vector_counts", {})
    fig, ax = plt.subplots(figsize=(8, 5))
    engines = list(counts.keys())
    values = [counts[e] for e in engines]
    colors = {"faiss": "#1565C0", "milvus": "#2E7D32", "pinecone": "#F57C00"}
    bars = ax.bar([e.upper() for e in engines], values,
                  color=[colors.get(e, "#607D8B") for e in engines])
    ax.set_ylabel("Vector Count")
    ax.set_title("Vector Index Build Summary")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(val),
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = output_dir / "charts" / "vector_index_bar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_tensorflow_results(tf_summary: dict, output_dir: Path) -> Path:
    if not tf_summary:
        return output_dir / "charts" / "keras_accuracy.png"

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
    ax.set_title("TensorFlow/Keras Production Model Comparison")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    path = output_dir / "charts" / "keras_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_summary_markdown(audit: dict, pytorch_summary: dict, tf_summary: dict,
                            vector_summary: dict) -> str:
    lines = [
        "# Sovereign Financial Auditing Engine — Pipeline Output",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Audit Report",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Status | {audit.get('status', 'N/A')} |",
        f"| Entities Audited | {audit.get('total_entities_audited', 0)} |",
        f"| High Risk Entities | {audit.get('high_risk_entities', 0)} ({audit.get('high_risk_percentage', 0)}%) |",
        f"| Average Risk Score | {audit.get('average_risk_score', 0)} |",
        "",
        "## PyTorch Models",
        "",
        "| Model | Val Loss | Val Accuracy |",
        "|-------|----------|--------------|",
    ]
    for name in ("cnn", "lstm", "transformer"):
        m = pytorch_summary.get(name, {})
        if m:
            lines.append(
                f"| {name.upper()} | {m.get('val_loss', 'N/A')} | {m.get('val_accuracy', 'N/A')} |"
            )

    opt = pytorch_summary.get("optimizer_comparison", {})
    if opt:
        lines.extend(["", "## Optimizer Comparison", ""])
        for name, data in opt.items():
            lines.append(f"- **{name.upper()}**: final_loss={data.get('final_loss', 'N/A')}")

    if tf_summary:
        lines.extend(["", "## TensorFlow/Keras Models", ""])
        for name, data in tf_summary.items():
            lines.append(
                f"- **{name.upper()}**: accuracy={data.get('accuracy', 'N/A')}, "
                f"backend={data.get('backend', 'N/A')}"
            )

    if vector_summary:
        counts = vector_summary.get("vector_counts", {})
        lines.extend([
            "",
            "## Vector Search Indices",
            "",
            f"- **FAISS:** {counts.get('faiss', 0)} vectors",
            f"- **Milvus:** {counts.get('milvus', 0)} vectors",
            f"- **Pinecone:** {counts.get('pinecone', 0)} vectors",
        ])

    lines.extend([
        "",
        "## Output Artifacts",
        "",
        "- `reports/` — JSON summaries and audit reports",
        "- `charts/` — Training curves, audit charts, vector index bar chart",
        "- `models/` — Saved model checkpoints",
        "- `logs/` — TensorBoard and training logs",
        "- `vector_indices/` — FAISS, Milvus, and Pinecone index artifacts",
    ])
    return "\n".join(lines)


def _copy_tree(src: Path, dst: Path, ignore_patterns: tuple[str, ...] = ()) -> list[str]:
    copied: list[str] = []
    if not src.exists():
        return copied
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        if any(item.match(p) for p in ignore_patterns):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied.append(str(rel))
    return copied


def _build_summary_text(audit: dict, pytorch_summary: dict, tf_summary: dict,
                        vector_summary: dict, elapsed_seconds: float | None = None) -> str:
    lines = [
        "Sovereign Financial Auditing Engine — Pipeline Output",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if elapsed_seconds is not None:
        lines.append(f"Pipeline elapsed: {elapsed_seconds:.1f}s")
    lines.extend([
        "",
        "AUDIT REPORT",
        f"  Status: {audit.get('status', 'N/A')}",
        f"  Entities Audited: {audit.get('total_entities_audited', 0)}",
        f"  High Risk: {audit.get('high_risk_entities', 0)} ({audit.get('high_risk_percentage', 0)}%)",
        f"  Average Risk Score: {audit.get('average_risk_score', 0)}",
        "",
        "PYTORCH MODELS",
    ])
    for name in ("cnn", "lstm", "transformer"):
        m = pytorch_summary.get(name, {})
        if m:
            lines.append(
                f"  {name.upper()}: val_loss={m.get('val_loss', 'N/A')}, "
                f"val_accuracy={m.get('val_accuracy', 'N/A')}"
            )

    opt = pytorch_summary.get("optimizer_comparison", {})
    if opt:
        lines.extend(["", "OPTIMIZER COMPARISON"])
        for name, data in opt.items():
            lines.append(f"  {name.upper()}: final_loss={data.get('final_loss', 'N/A')}")

    if tf_summary:
        lines.extend(["", "TENSORFLOW/KERAS MODELS"])
        for name, data in tf_summary.items():
            lines.append(
                f"  {name.upper()}: accuracy={data.get('accuracy', 'N/A')}, "
                f"backend={data.get('backend', 'N/A')}"
            )

    if vector_summary:
        counts = vector_summary.get("vector_counts", {})
        lines.extend([
            "",
            "VECTOR SEARCH INDICES",
            f"  FAISS: {counts.get('faiss', 0)} vectors",
            f"  Milvus: {counts.get('milvus', 0)} vectors",
            f"  Pinecone: {counts.get('pinecone', 0)} vectors",
        ])
    return "\n".join(lines)


def collect_all_outputs(config: dict | None = None, elapsed_seconds: float | None = None) -> Path:
    """Gather all pipeline artifacts into output/ directory."""
    config = config or load_config()
    output_dir = Path(config["paths"]["output"])
    for sub in ("reports", "charts", "data_samples", "models", "logs", "vector_indices"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    models_dir = Path(config["paths"]["models"])
    sample_dir = Path(config["paths"]["data_sample"])
    logs_dir = Path(config["paths"]["logs"])
    vector_dir = Path(config["paths"]["vector_indices"])
    processed_dir = Path(config["paths"]["data_processed"])

    pytorch_summary = _load_json(models_dir / "pytorch_training_summary.json") or {}
    tf_summary = _load_json(models_dir / "tensorflow_training_summary.json") or {}
    audit = _load_json(processed_dir / "audit_report.json") or {}
    vector_summary = _load_json(vector_dir / "build_summary.json") or {}

    pipeline_report = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "project": config.get("project", {}),
        "audit_status": audit.get("status"),
        "entities_audited": audit.get("total_entities_audited"),
        "high_risk_entities": audit.get("high_risk_entities"),
        "pytorch_models": {k: v for k, v in pytorch_summary.items() if k != "optimizer_comparison"},
        "optimizer_best": min(
            (pytorch_summary.get("optimizer_comparison") or {}),
            key=lambda k: (pytorch_summary["optimizer_comparison"][k].get("final_loss", 999)),
            default="adam",
        ),
        "tensorflow_models": tf_summary,
        "vector_indices": vector_summary,
    }

    with open(output_dir / "reports" / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_report, f, indent=2)

    with open(output_dir / "reports" / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    report_copies = ["pytorch_training_summary.json", "tensorflow_training_summary.json", "build_summary.json"]
    for name in report_copies:
        src = models_dir / name if name != "build_summary.json" else vector_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / "reports" / name)

    summary_md = _build_summary_markdown(audit, pytorch_summary, tf_summary, vector_summary)
    summary_txt = _build_summary_text(
        audit, pytorch_summary, tf_summary, vector_summary, elapsed_seconds
    )
    with open(output_dir / "pipeline_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
    with open(output_dir / "pipeline_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)

    chart_paths: list[str] = []
    if pytorch_summary:
        chart_paths.append(str(_plot_training_curves(pytorch_summary, output_dir)))
        opt_data = pytorch_summary.get("optimizer_comparison")
        if opt_data:
            chart_paths.append(str(_plot_optimizer_comparison(opt_data, output_dir)))
    if audit:
        chart_paths.append(str(_plot_audit_charts(audit, output_dir)))
    if tf_summary:
        chart_paths.append(str(_plot_tensorflow_results(tf_summary, output_dir)))
    if vector_summary:
        chart_paths.append(str(_plot_vector_index_bar_chart(vector_summary, output_dir)))

    copied_samples: list[str] = []
    for src in sample_dir.glob("*"):
        if src.is_file():
            shutil.copy2(src, output_dir / "data_samples" / src.name)
            copied_samples.append(f"data_samples/{src.name}")

    copied_models: list[str] = []
    for pattern in ("*.pt", "*.json", "tensorflow_*"):
        for src in models_dir.glob(pattern):
            if src.name.endswith("_summary.json") or src.name.endswith(".backend.json"):
                continue
            if src.is_dir():
                copied_models.extend(_copy_tree(src, output_dir / "models" / src.name))
            else:
                shutil.copy2(src, output_dir / "models" / src.name)
                copied_models.append(src.name)

    copied_logs = _copy_tree(logs_dir, output_dir / "logs")
    copied_indices = _copy_tree(vector_dir, output_dir / "vector_indices", ignore_patterns=("*.prev",))

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "output_dir": str(output_dir),
        "reports": [
            "reports/pipeline_summary.json",
            "reports/audit_report.json",
            "pipeline_summary.txt",
        ],
        "charts": [str(Path(p).relative_to(output_dir)) for p in chart_paths],
        "data_samples": copied_samples,
        "models": copied_models,
        "logs": copied_logs,
        "vector_indices": copied_indices,
        "audit_status": audit.get("status"),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(summary_txt)
    print(f"\nManifest written to: {output_dir / 'manifest.json'}")
    print(f"All outputs saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    collect_all_outputs()
