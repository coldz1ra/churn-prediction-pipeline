from src.utils import ensure_binary_series
import pandas as pd
import pytest

def test_ensure_binary_series_ok():
    s = pd.Series([0,1,1,0,0])
    out = ensure_binary_series(s)
    assert set(out.unique().tolist()) == {0,1}

def test_ensure_binary_series_fail():
    s = pd.Series([0,1,2])
    with pytest.raises(ValueError):
        ensure_binary_series(s)
