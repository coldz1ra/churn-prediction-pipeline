from __future__ import annotations
import argparse, json, yaml, os
from datetime import datetime
import pandas as pd

def _try_get_tree_feature_importance(model):
    # LightGBM / XGB: feature_importances_ or get_booster().get_score()
    try:
        import numpy as np
        import pandas as pd
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        # Calibrated models
        if hasattr(model, "base_estimator") and hasattr(model.base_estimator, "feature_importances_"):
            return model.base_estimator.feature_importances_
        # XGB booster
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            score_dict = booster.get_score(importance_type="gain")
            # Return aligned later in calling code
            return score_dict
        if hasattr(model, "base_estimator") and hasattr(model.base_estimator, "get_booster"):
            booster = model.base_estimator.get_booster()
            score_dict = booster.get_score(importance_type="gain")
            return score_dict
    except Exception:
        pass
    return None

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Metrics
    metrics = {}
    if os.path.exists("reports/metrics.json"):
        metrics = json.load(open("reports/metrics.json", "r", encoding="utf-8"))

    # Validation predictions (has engineered features, y_true/proba)
    val_df = None
    if os.path.exists("reports/val_predictions.csv"):
        val_df = pd.read_csv("reports/val_predictions.csv")

    # Config numbers
    kf = cfg.get("thresholding", {}).get("k_fraction", 0.1)
    cps = cfg.get("thresholding", {}).get("cps", 2.0)
    ltv = cfg.get("thresholding", {}).get("ltv", 100.0)

    # Business insights calculation
    insights = {}
    if val_df is not None and "y_proba" in val_df and "y_true" in val_df:
        n = len(val_df)
        k = max(1, int(n * kf))
        val_sorted = val_df.sort_values("y_proba", ascending=False).reset_index(drop=True)
        baseline = val_df["y_true"].mean()
        top_hit_rate = val_sorted.head(k)["y_true"].mean()
        lift = (top_hit_rate / (baseline + 1e-12))

        # expected saved and profit proxy
        expected_saves = float(val_sorted.head(k)["y_proba"].sum())
        cost = k * cps
        expected_revenue = expected_saves * ltv
        profit_proxy = expected_revenue - cost

        insights = {
            "n_val": int(n),
            "k_fraction": float(kf),
            "k_count": int(k),
            "baseline_rate": float(baseline),
            "top_hit_rate": float(top_hit_rate),
            "lift@k": float(lift),
            "expected_saves@k": expected_saves,
            "cost@k": float(cost),
            "ltv": float(ltv),
            "expected_profit_proxy@k": float(profit_proxy),
        }

        # add expected_saving column in topk_contacts.csv if exists
        topk_path = "reports/topk_contacts.csv"
        if os.path.exists(topk_path):
            topk = pd.read_csv(topk_path)
            if "y_proba" in topk.columns:
                topk["expected_saving"] = topk["y_proba"] * ltv
                topk.to_csv(topk_path, index=False)

    # Try to compute SHAP top-10 if possible (optional)
    shap_top = None
    try:
        import shap, joblib, json
        import numpy as np
        model = joblib.load("artifacts/model.joblib")
        X_val = val_df.drop(columns=["y_true","y_proba"]) if val_df is not None else None
        base_est = model
        if hasattr(model, "base_estimator"):
            base_est = model.base_estimator
        if X_val is not None and ("LGBM" in type(base_est).__name__ or "XGB" in type(base_est).__name__):
            explainer = shap.TreeExplainer(base_est)
            vals = explainer.shap_values(X_val)
            if isinstance(vals, list):
                # binary: pick positive class
                vals = vals[1] if len(vals) > 1 else vals[0]
            mean_abs = np.abs(vals).mean(axis=0)
            shap_top = sorted(
                [
                    (feat, float(mean_abs[i]))
                    for i, feat in enumerate(X_val.columns.tolist())
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10]
    except Exception:
        shap_top = None

    # Tree feature importance (gain or split)
    feat_imp = None
    try:
        import joblib
        model = joblib.load("artifacts/model.joblib")
        imp = _try_get_tree_feature_importance(model)
        if isinstance(imp, dict) and val_df is not None:
            # XGB booster dict; align to columns if possible
            cols = [c for c in val_df.columns if c not in ("y_true","y_proba")]
            feat_imp = sorted([(k, imp.get(k, 0.0)) for k in imp.keys()], key=lambda x: x[1], reverse=True)[:10]
        elif imp is not None and val_df is not None:
            cols = [c for c in val_df.columns if c not in ("y_true","y_proba")]
            feat_imp = sorted(list(zip(cols, imp)), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        feat_imp = None

    lines = []
    lines.append(f"# Customer Churn — Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append("## Overview")
    lines.append("- End-to-end pipeline with calibrated probabilities, business lift and profit simulation.")
    lines.append("- Models: Logistic Regression / LightGBM / XGBoost (with early stopping).")
    lines.append("")

    if metrics:
        lines.append("## Metrics")
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if insights:
        lines.append("## Business Insights")
        lines.append(f"- Top-{int(insights['k_fraction']*100)}% contact list size: **{insights['k_count']}**.")
        lines.append(f"- Baseline churn rate: **{insights['baseline_rate']:.3f}**; Top-k hit rate: **{insights['top_hit_rate']:.3f}**; Lift@k: **{insights['lift@k']:.2f}×**.")
        lines.append(f"- Expected saves@k: **{insights['expected_saves@k']:.1f}**; Cost@k (CPS={cps}): **{insights['cost@k']:.2f}**.")
        lines.append(f"- LTV assumption: **{ltv}**; Expected profit proxy@k: **{insights['expected_profit_proxy@k']:.2f}**.")
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
    for fig in ["roc_curve.png","pr_curve.png","calibration_curve.png","lift_curve.png","profit_curve.png","shap_summary.png"]:
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
