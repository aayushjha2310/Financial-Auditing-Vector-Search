"""Risk analytics and predictive risk mitigation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RiskAssessment:
    """Single transaction/account risk assessment."""

    entity_id: str
    risk_score: float
    anomaly_flag: bool
    volatility: float
    z_score: float
    recommendation: str


@dataclass
class PortfolioRiskReport:
    """Aggregate portfolio-level risk report."""

    total_entities: int
    high_risk_count: int
    avg_risk_score: float
    max_volatility: float
    assessments: list[RiskAssessment] = field(default_factory=list)
    currency_exposure: dict[str, float] = field(default_factory=dict)


class RiskAnalyticsEngine:
    """
    Predictive risk mitigation engine combining deep learning outputs
    with statistical financial auditing heuristics.
    """

    def __init__(self, anomaly_threshold: float = 0.75, volatility_window: int = 20):
        self.anomaly_threshold = anomaly_threshold
        self.volatility_window = volatility_window

    def compute_statistical_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistical risk metrics on processed ledger."""
        result = df.copy()
        if "z_score" not in result.columns:
            grouped = result.groupby(["currency", "account_id"], observed=True)
            result["rolling_vol"] = grouped["net_flow"].transform(
                lambda x: x.rolling(self.volatility_window, min_periods=1).std().fillna(0)
            )
            result["rolling_mean"] = grouped["net_flow"].transform(
                lambda x: x.rolling(self.volatility_window, min_periods=1).mean().fillna(0)
            )
            result["z_score"] = (
                (result["net_flow"] - result["rolling_mean"])
                / (result["rolling_vol"] + 1e-8)
            )
        result["stat_risk_score"] = (
            0.4 * result["z_score"].abs().clip(0, 3) / 3
            + 0.3 * (result.get("rolling_vol", 0) / (result.get("rolling_vol", pd.Series([1])).max() + 1e-8))
            + 0.3 * result.get("risk_score", 0.5)
        ).clip(0, 1)
        return result

    def merge_model_predictions(
        self,
        df: pd.DataFrame,
        cls_probs: np.ndarray,
        reg_scores: np.ndarray,
        stat_weight: float = 0.4,
        cls_weight: float = 0.4,
        reg_weight: float = 0.2,
    ) -> pd.DataFrame:
        """Fuse statistical, classification, and regression model outputs."""
        result = df.copy()
        n = min(len(result), len(cls_probs), len(reg_scores))
        result = result.iloc[:n].copy()
        result["model_cls_prob"] = cls_probs[:n]
        result["model_reg_score"] = reg_scores[:n]
        if "stat_risk_score" not in result.columns:
            result = self.compute_statistical_risk(result)
        result["combined_risk_score"] = (
            stat_weight * result["stat_risk_score"]
            + cls_weight * result["model_cls_prob"]
            + reg_weight * result["model_reg_score"].clip(0, 1)
        ).clip(0, 1)
        result["anomaly_flag"] = result["combined_risk_score"] >= self.anomaly_threshold
        return result

    def generate_assessments(self, df: pd.DataFrame) -> list[RiskAssessment]:
        """Generate per-entity risk assessments with recommendations."""
        assessments = []
        for _, row in df.iterrows():
            score = row.get("combined_risk_score", row.get("stat_risk_score", 0.5))
            z = row.get("z_score", 0.0)
            vol = row.get("rolling_volatility", row.get("rolling_vol", 0.0))
            flagged = bool(row.get("anomaly_flag", score >= self.anomaly_threshold))

            if score >= 0.9:
                rec = "IMMEDIATE REVIEW: Escalate to compliance officer"
            elif score >= 0.75:
                rec = "HIGH RISK: Initiate enhanced due diligence"
            elif score >= 0.5:
                rec = "MODERATE: Schedule periodic audit review"
            else:
                rec = "LOW RISK: Standard monitoring sufficient"

            entity_id = f"{row.get('account_id', 'unknown')}_{row.get('currency', 'NA')}"
            assessments.append(RiskAssessment(
                entity_id=str(entity_id),
                risk_score=float(score),
                anomaly_flag=flagged,
                volatility=float(vol),
                z_score=float(z),
                recommendation=rec,
            ))
        return assessments

    def generate_portfolio_report(self, df: pd.DataFrame) -> PortfolioRiskReport:
        """Generate aggregate portfolio risk report."""
        if "combined_risk_score" not in df.columns:
            df = self.compute_statistical_risk(df)

        assessments = self.generate_assessments(df)
        high_risk = sum(1 for a in assessments if a.anomaly_flag)

        currency_exposure = {}
        if "currency" in df.columns and "balance_base" in df.columns:
            currency_exposure = df.groupby("currency", observed=True)["balance_base"].sum().to_dict()

        return PortfolioRiskReport(
            total_entities=len(assessments),
            high_risk_count=high_risk,
            avg_risk_score=float(df.get("combined_risk_score", df.get("stat_risk_score", pd.Series([0.5]))).mean()),
            max_volatility=float(df.get("rolling_volatility", df.get("rolling_vol", pd.Series([0.0]))).max()),
            assessments=assessments[:100],
            currency_exposure=currency_exposure,
        )

    def audit_summary(self, report: PortfolioRiskReport) -> dict[str, Any]:
        """Format audit summary for compliance reporting."""
        return {
            "status": "CRITICAL" if report.high_risk_count > report.total_entities * 0.1 else "NORMAL",
            "total_entities_audited": report.total_entities,
            "high_risk_entities": report.high_risk_count,
            "high_risk_percentage": round(100 * report.high_risk_count / max(report.total_entities, 1), 2),
            "average_risk_score": round(report.avg_risk_score, 4),
            "max_volatility": round(report.max_volatility, 4),
            "currency_exposure": {k: round(v, 2) for k, v in report.currency_exposure.items()},
            "top_risks": [
                {
                    "entity": a.entity_id,
                    "score": round(a.risk_score, 4),
                    "recommendation": a.recommendation,
                }
                for a in sorted(report.assessments, key=lambda x: x.risk_score, reverse=True)[:10]
            ],
        }
