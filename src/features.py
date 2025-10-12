from __future__ import annotations

import pandas as pd


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Простые агрегаты
    if {"revenue", "active_days"}.issubset(df.columns):
        df = df.copy()
        df["arpu"] = df["revenue"] / df["active_days"].clip(lower=1)
    if "tenure_days" in df.columns:
        df["tenure_months"] = df["tenure_days"] / 30.0

    # One-hot для object-колонок
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
