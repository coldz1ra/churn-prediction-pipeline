from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import basic_features
from src.plots import plot_lift, plot_pr, plot_roc
from src.utils import ensure_binary_series

TREE_TYPES = {"xgb", "lgbm"}


def build_model(cfg):
    mtype = cfg["model"]["type"]
    if mtype == "logreg":
        from sklearn.linear_model import LogisticRegression

        params = cfg["model"]["params"]
        return LogisticRegression(**params)
    elif mtype == "xgb":
        from xgboost import XGBClassifier

        params = cfg["model"].get("params", {})
        return XGBClassifier(eval_metric="logloss", tree_method="hist", **params)
    elif mtype == "lgbm":
        from lightgbm import LGBMClassifier

        params = cfg["model"].get("params", {})
        return LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def fit_with_optional_early_stopping(model, mtype: str, X_tr, y_tr, X_val, y_val):
    if mtype == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)
    elif mtype == "lgbm":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
    else:
        model.fit(X_tr, y_tr)
    return model


def main(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data"]["train_path"])
    y = ensure_binary_series(df[cfg["data"]["target"]])
    id_col = cfg["data"]["id_col"]
    X_raw = df.drop(columns=[cfg["data"]["target"], id_col, "Churn"], errors="ignore")
    X = basic_features(X_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg["data"]["valid_size"],
        stratify=y,
        random_state=cfg["data"]["random_state"],
    )

    ros = RandomOverSampler(random_state=cfg["data"]["random_state"])
    X_tr, y_tr = ros.fit_resample(X_train, y_train)

    mtype = cfg["model"]["type"]
    model = build_model(cfg)
    calib = cfg.get("calibration", {}).get("method", None)

    if calib and mtype in TREE_TYPES:
        model_prefit = fit_with_optional_early_stopping(model, mtype, X_tr, y_tr, X_val, y_val)
        model = CalibratedClassifierCV(estimator=model_prefit, method=calib, cv="prefit")
        model.fit(X_val, y_val)
    else:
        model = fit_with_optional_early_stopping(model, mtype, X_tr, y_tr, X_val, y_val)
        if calib and mtype not in TREE_TYPES:
            model = CalibratedClassifierCV(estimator=model, method=calib, cv=3).fit(X_tr, y_tr)

    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    print(f"ROC-AUC: {auc:.4f}")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    out = X_val.copy()
    out["y_true"] = y_val.values
    out["y_proba"] = proba
    out.to_csv("reports/val_predictions.csv", index=False)

    plot_roc(y_val, proba, "reports/figures/roc_curve.png")
    plot_pr(y_val, proba, "reports/figures/pr_curve.png")
    plot_lift(y_val, proba, "reports/figures/lift_curve.png")

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")
    with open("artifacts/columns.json", "w", encoding="utf-8") as f:
        json.dump({"columns": X.columns.tolist()}, f)

    k_frac = cfg.get("thresholding", {}).get("k_fraction", 0.1)
    n = len(out)
    k = max(1, int(n * k_frac))
    recs = out.sort_values("y_proba", ascending=False).head(k).copy()
    recs.to_csv("reports/topk_contacts.csv", index=False)

    metrics = {"roc_auc": float(auc), "n_val": int(len(y_val)), "k_fraction": float(k_frac)}
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    print("Artifacts saved: artifacts/*, reports/val_predictions.csv, ")
    print("reports/topk_contacts.csv, reports/metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
