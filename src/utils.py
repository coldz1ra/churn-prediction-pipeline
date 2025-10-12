from __future__ import annotations
import numpy as np
import pandas as pd

def ensure_binary_series(y: pd.Series) -> pd.Series:
    uniq = set(pd.Series(y).dropna().unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError("Target must be binary 0/1")
    return y.astype(int)

def train_valid_split(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
