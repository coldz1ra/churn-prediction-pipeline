import pandas as pd
from src.features import basic_features

def test_basic_features_creates_arpu_and_tenure():
    df = pd.DataFrame({
        "revenue":[100, 0],
        "active_days":[10, 1],
        "tenure_days":[60, 30],
        "cat":["A","B"]
    })
    out = basic_features(df)
    assert "arpu" in out.columns
    assert "tenure_months" in out.columns
    # One-hot
    assert any(c.startswith("cat_") for c in out.columns)
