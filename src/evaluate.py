from __future__ import annotations
import argparse
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from src.metrics import compute_core_metrics, lift_at_k
from src.thresholds import select_by_k_fraction, profit_curve

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv("reports/val_predictions.csv")
    y_true = df["y_true"]
    y_proba = df["y_proba"].values

    # Metrics
    core = compute_core_metrics(y_true, y_proba)
    kf = cfg.get("thresholding", {}).get("k_fraction", 0.1)
    core["lift@k"] = lift_at_k(y_true, y_proba, kf)
    thr = select_by_k_fraction(y_proba, kf)
    core["threshold@k"] = float(thr)
    print(core)

    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(core, f, ensure_ascii=False, indent=2)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", linestyle="-", label="Model")
    plt.plot([0,1],[0,1], linestyle="--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/calibration_curve.png")

    # Profit curve (simple)
    cps = cfg.get("thresholding", {}).get("cps", 2.0)
    prof = profit_curve(y_true, y_proba, cps=cps)
    pr_df = pd.DataFrame(prof)
    pr_df.to_csv("reports/profit_curve.csv", index=False)
    plt.figure()
    plt.plot(pr_df["k_fraction"], pr_df["saved"] - pr_df["cost"], marker="o")
    plt.xlabel("Top fraction")
    plt.ylabel("Saved - Cost (proxy)")
    plt.title("Profit Curve (proxy)")
    plt.tight_layout()
    plt.savefig("reports/figures/profit_curve.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
