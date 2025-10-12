from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def compute_core_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "logloss": float(log_loss(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def lift_at_k(y_true: pd.Series, y_proba: np.ndarray, k_frac: float = 0.1) -> float:
    n = len(y_true)
    k = max(1, int(n * k_frac))
    order = np.argsort(-y_proba)[:k]
    top_k = np.array(y_true)[order]
    baseline_rate = np.mean(y_true)
    return float(np.mean(top_k) / (baseline_rate + 1e-12))
