from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _try_shap_top(
    model: Any, X_val: Optional[pd.DataFrame], topn: int = 10
) -> Optional[List[Tuple[str, float]]]:
    try:
        import numpy as np
        import shap  # type: ignore

        base = getattr(model, "estimator", getattr(model, "base_estimator", model))
        if X_val is None:
            return None
        if "LGBM" not in type(base).__name__ and "XGB" not in type(base).__name__:
            return None
        explainer = shap.TreeExplainer(base)
        vals = explainer.shap_values(X_val)
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        mean_abs = np.abs(vals).mean(axis=0)
        names = X_val.columns.tolist()
        pairs = sorted(
            [(names[i], float(mean_abs[i])) for i in range(len(names))],
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs[:topn]
    except Exception:
        return None


def _try_tree_importance(
    model: Any, feature_names: List[str], topn: int = 10
) -> Optional[List[Tuple[str, float]]]:
    try:
        base = getattr(model, "estimator", getattr(model, "base_estimator", model))
        if hasattr(base, "feature_importances_"):
            imp = list(base.feature_importances_)
            pairs = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:topn]
            return [(k, float(v)) for k, v in pairs]
        if hasattr(base, "get_booster"):
            booster = base.get_booster()
            score = booster.get_score(importance_type="gain")
            pairs = sorted(score.items(), key=lambda x: x[1], reverse=True)[:topn]
            return [(k, float(v)) for k, v in pairs]
    except Exception:
        return None
    return None


def main(cfg_path: str) -> None:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metrics = _load_json("reports/metrics.json") or {}
    val_df = None
    if os.path.exists("reports/val_predictions.csv"):
        val_df = pd.read_csv("reports/val_predictions.csv")

    kf = float(cfg.get("thresholding", {}).get("k_fraction", 0.1))
    cps = float(cfg.get("thresholding", {}).get("cps", 2.0))
    ltv = float(cfg.get("thresholding", {}).get("ltv", 100.0))

    insights: Dict[str, float] = {}
    if val_df is not None and {"y_true", "y_proba"}.issubset(val_df.columns):
        n = len(val_df)
        k = max(1, int(n * kf))
        baseline = float(val_df["y_true"].mean())
        top = val_df.sort_values("y_proba", ascending=False).head(k)
        top_hit = float(top["y_true"].mean())
        lift = top_hit / (baseline + 1e-12)
        expected_saves = float(top["y_proba"].sum())
        cost = float(k * cps)
        expected_profit = expected_saves * ltv - cost
        insights = {
            "n_val": float(n),
            "k_fraction": kf,
            "k_count": float(k),
            "baseline_rate": baseline,
            "top_hit_rate": top_hit,
            "lift@k": float(lift),
            "expected_saves@k": expected_saves,
            "cost@k": cost,
            "ltv": ltv,
            "expected_profit_proxy@k": float(expected_profit),
        }
        topk_path = "reports/topk_contacts.csv"
        if os.path.exists(topk_path):
            topk = pd.read_csv(topk_path)
            if "y_proba" in topk.columns:
                topk["expected_saving"] = topk["y_proba"] * ltv
                topk.to_csv(topk_path, index=False)

    model = None
    try:
        import joblib  # type: ignore

        model = joblib.load("artifacts/model.joblib")
    except Exception:
        model = None

    X_val = None
    if val_df is not None:
        X_val = val_df.drop(columns=["y_true", "y_proba"], errors="ignore")

    shap_top = _try_shap_top(model, X_val, topn=10) if model is not None else None
    feat_imp = (
        _try_tree_importance(model, X_val.columns.tolist(), topn=10)
        if (model is not None and X_val is not None)
        else None
    )

    lines: List[str] = []
    lines.append("# Customer Churn — Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append("## Overview")
    lines.append("- Calibrated probabilities, lift@k and profit simulation on the Telco dataset.")
    lines.append("- Models: Logistic Regression / LightGBM / XGBoost.")
    lines.append("")

    if metrics:
        lines.append("## Metrics")
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if insights:
        lines.append("## Business Insights")
        lines.append(
            f"- Top-{int(insights['k_fraction'] * 100)}% contact list size: "
            f"**{int(insights['k_count'])}**."
        )
        lines.append(
            f"- Baseline churn rate: **{insights['baseline_rate']:.3f}**; "
            f"Top-k hit rate: **{insights['top_hit_rate']:.3f}**."
        )
        lines.append(f"- Lift@k: **{insights['lift@k']:.2f}×**.")
        lines.append(
            f"- Expected saves@k: **{insights['expected_saves@k']:.1f}**; "
            f"Cost@k (CPS={cps}): **{insights['cost@k']:.2f}**."
        )
        lines.append(
            f"- LTV: **{ltv:.0f}**; "
            f"Expected profit proxy@k: **{insights['expected_profit_proxy@k']:.2f}**."
        )
        lines.append("")

    if shap_top:
        lines.append("## Top 10 SHAP Features (mean |SHAP|)")
        for name, val in shap_top:
            lines.append(f"- {name}: {val:.6f}")
        lines.append("")
    if feat_imp:
        lines.append("## Top 10 Tree Feature Importances")
        for name, val in feat_imp:
            lines.append(f"- {name}: {val:.6f}")
        lines.append("")

    lines.append("## Figures")
    figs = [
        "roc_curve.png",
        "pr_curve.png",
        "calibration_curve.png",
        "lift_curve.png",
        "profit_curve.png",
        "shap_summary.png",
    ]
    for fig in figs:
        p = f"reports/figures/{fig}"
        if os.path.exists(p):
            lines.append(f"![{fig}]({p})")

    with open("reports/churn_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved reports/churn_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
