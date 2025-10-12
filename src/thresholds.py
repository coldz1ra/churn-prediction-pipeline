from __future__ import annotations
import numpy as np
import pandas as pd

def select_by_k_fraction(y_proba, k_fraction: float):
    n = len(y_proba)
    k = max(1, int(n * k_fraction))
    thr = np.partition(y_proba, -k)[-k]
    return float(thr)

def profit_curve(y_true, y_proba, cps: float = 2.0, k_grid=None):
    # Пример: считаем относительную выгоду при контакте top-k%
    if k_grid is None:
        k_grid = [i / 100 for i in range(1, 51)]
    out = []
    order = np.argsort(-y_proba)
    y_sorted = np.array(y_true)[order]
    for kf in k_grid:
        k = max(1, int(len(y_true) * kf))
        contacted = y_sorted[:k]
        saved = contacted.sum()  # упрощённая метрика
        cost = k * cps
        out.append({"k_fraction": kf, "saved": int(saved), "cost": float(cost)})
    return out
