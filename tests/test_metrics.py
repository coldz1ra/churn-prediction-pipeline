import numpy as np
import pandas as pd
from src.metrics import lift_at_k

def test_lift_at_k_reasonable():
    y = pd.Series([0,1,0,1,0,0,1,0,0,0])
    proba = np.array([0.1,0.9,0.2,0.8,0.3,0.2,0.7,0.1,0.05,0.4])
    lift = lift_at_k(y, proba, 0.3)  # top-3
    assert lift > 1.0
