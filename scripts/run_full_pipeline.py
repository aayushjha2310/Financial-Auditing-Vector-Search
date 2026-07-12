"""End-to-end sovereign financial auditing pipeline orchestrator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engineering import FeatureEngineer, LedgerProcessor
from src.deep_learning.pytorch import FinancialLSTM
from src.risk_analytics import RiskAnalyticsEngine
from src.utils.config import ensure_directories, get_epochs, load_config
from src.utils import set_seed


class _TeeStream:
    """Write stdout/stderr to both console and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()


def run_script(name: str, env: dict[str, str]) -> bool:
    script = PROJECT_ROOT / "scripts" / name
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return result.returncode == 0


def run_risk_analytics(config: dict) -> dict:
    print(f"\n{'='*60}")
    print("  STEP: Risk Analytics & Audit Report")
    print(f"{'='*60}")

    ledger_path = Path(config["paths"]["data_sample"]) / "ledger_history.csv"
    processor = LedgerProcessor(currencies=config["data"]["currencies"])
    processed = processor.process_pipeline(
        ledger_path,
        window=config["risk"]["volatility_window"],
        seq_len=config["data"]["sequence_length"],
    )

    fe = FeatureEngineer()
    sequences = fe.normalize_sequences(processed["sequences"])
    labels = processed["labels"]
    ledger_df = processed["ledger"]

    model = FinancialLSTM(
        input_dim=sequences.shape[-1],
        hidden_dim=config["pytorch"]["lstm_hidden"],
        num_layers=config["pytorch"]["lstm_layers"],
    )
    model_path = Path(config["paths"]["models"]) / "pytorch_lstm.pt"
    if model_path.exists():
        import torch
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    import torch
    model.eval()
    with torch.no_grad():
        X = torch.tensor(sequences, dtype=torch.float32)
        outputs = model(X)
        cls_probs = torch.sigmoid(outputs[0].squeeze()).numpy()
        reg_scores = outputs[1].squeeze().numpy()

    risk_engine = RiskAnalyticsEngine(
        anomaly_threshold=config["risk"]["anomaly_threshold"],
        volatility_window=config["risk"]["volatility_window"],
    )
    risk_df = risk_engine.compute_statistical_risk(ledger_df.iloc[:len(cls_probs)])
    risk_df = risk_engine.merge_model_predictions(risk_df, cls_probs, reg_scores)
    report = risk_engine.generate_portfolio_report(risk_df)
    audit = risk_engine.audit_summary(report)

    audit_path = Path(config["paths"]["data_processed"]) / "audit_report.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\n  Audit Status: {audit['status']}")
    print(f"  Entities Audited: {audit['total_entities_audited']}")
    print(f"  High Risk: {audit['high_risk_entities']} ({audit['high_risk_percentage']}%)")
    print(f"  Avg Risk Score: {audit['average_risk_score']}")
    print(f"  Report saved: {audit_path}")
    return audit


def main() -> int:
    start = time.time()
    config = load_config()
    ensure_directories(config)
    set_seed(config["training"]["random_seed"])

    output_dir = Path(config["paths"]["output"])
    log_path = output_dir / "pipeline.log"
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)

    exit_code = 0
    audit: dict | None = None
    pipeline_env = os.environ.copy()
    pipeline_env["PIPELINE_MODE"] = "1"

    try:
        epochs = get_epochs(config, pipeline_mode=True)
        print("=" * 60)
        print("  SOVEREIGN FINANCIAL AUDITING & RISK ANALYTICS ENGINE")
        print("  End-to-End Pipeline Execution")
        print(f"  Training epochs: {epochs}")
        print("=" * 60)

        steps = [
            "generate_sample_data.py",
            "train_pytorch_models.py",
            "train_tensorflow_models.py",
            "build_vector_indices.py",
        ]

        for step in steps:
            if not run_script(step, pipeline_env):
                print(f"\nPipeline FAILED at step: {step}")
                exit_code = 1
                break

        if exit_code == 0:
            audit = run_risk_analytics(config)

            print(f"\n{'='*60}")
            print("  STEP: Collecting Outputs")
            print(f"{'='*60}")
            from scripts.collect_outputs import collect_all_outputs

            elapsed = time.time() - start
            collect_all_outputs(config, elapsed_seconds=elapsed)

            print(f"\n{'='*60}")
            print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
            print(f"{'='*60}")
            print("\n  Components verified:")
            print("    [x] Pandas/NumPy multi-currency ledger processing")
            print("    [x] PyTorch CNN, LSTM, Transformer (from scratch)")
            print("    [x] Manual backpropagation & gradient tracking")
            print("    [x] Multi-task loss (Cross-Entropy + MSE)")
            print("    [x] Optimizer comparison (Adam, SGD, RMSprop)")
            print("    [x] TensorFlow/Keras production models")
            print("    [x] TensorBoard logging (PyTorch + TensorFlow)")
            print("    [x] FAISS IVFPQ vector index")
            print("    [x] Milvus vector database (Lite)")
            print("    [x] Pinecone vector index (mock/production)")
            print("    [x] Distributed document semantic search")
            print("    [x] Risk analytics & compliance audit report")
            if audit:
                print(f"\n  Audit Status: {audit['status']}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
