import numpy as np

from src.thresholds import select_by_k_fraction


def test_select_by_k_fraction_topk_threshold():
    proba = np.array([0.1, 0.8, 0.3, 0.6, 0.2])
    thr = select_by_k_fraction(proba, 0.4)  # top-2
    top_mask = proba >= thr
    assert top_mask.sum() >= 2
