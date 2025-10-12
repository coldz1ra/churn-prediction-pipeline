import numpy as np
import pandas as pd
from src.thresholds import profit_curve

def test_profit_curve_monotonic_cost():
    y = pd.Series([0,1,0,1,0,0,1,0,0,0])
    proba = np.linspace(0.1,0.9,10)
    curve = profit_curve(y, proba, cps=2.0, k_grid=[0.1,0.2,0.3,0.4,0.5])
    costs = [pt["cost"] for pt in curve]
    assert costs == sorted(costs), "Cost should increase with k"
