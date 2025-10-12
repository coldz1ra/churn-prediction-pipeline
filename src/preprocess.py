from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Явно разделяем типы признаков
def split_columns(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocess(df_sample: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = split_columns(df_sample)
    numeric = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre
