from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def plot_roc(y_true, y_proba, path: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)

def plot_pr(y_true, y_proba, path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)

def plot_lift(y_true, y_proba, path: str, bins: int = 10):
    order = np.argsort(-y_proba)
    y_sorted = np.array(y_true)[order]
    baseline = y_sorted.mean()
    lifts = []
    xs = []
    n = len(y_sorted)
    for i in range(1, bins + 1):
        k = int(n * (i / bins))
        xs.append(i / bins)
        lifts.append(y_sorted[:k].mean() / (baseline + 1e-12))
    plt.figure()
    plt.plot(xs, lifts, marker="o")
    plt.xlabel("Top fraction of customers")
    plt.ylabel("Lift")
    plt.title("Cumulative Lift Curve")
    plt.tight_layout()
    plt.savefig(path)
