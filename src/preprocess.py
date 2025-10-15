from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ToNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(pd.to_numeric, errors="coerce").to_numpy()
        # ndarray -> через DataFrame, чтобы корректно coerce'ить пустые строки
        return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy()


class ToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.astype("string")
        return pd.DataFrame(X).astype("string")


def split_columns(df: pd.DataFrame):
    num_cols = df.select_dtypes(
        include=["int64", "float64", "int32", "float32", "bool"]
    ).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    # Особый случай Telco: TotalCharges должен быть числом
    if "TotalCharges" in cat_cols:
        as_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if as_num.notna().any():
            num_cols.append("TotalCharges")
            cat_cols.remove("TotalCharges")
    return num_cols, cat_cols


def build_preprocess(df_sample: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = split_columns(df_sample)

    numeric = Pipeline(
        steps=[
            ("to_numeric", ToNumeric()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("to_string", ToString()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre
