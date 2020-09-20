import numpy as np


def test_get_gini_score():
    """A quick example against a hand worked gini score."""
    from my_ml.model.decision_tree import DecisionTree

    dtree = DecisionTree()
    low_y: np.ndarray = np.array([1, 0, 0])
    high_y: np.ndarray = np.array([1, 0])
    k: int = 2
    gini_score: float = dtree._get_gini_score(low_y, high_y, k)
    assert np.isclose(0.47, gini_score, rtol=0.008)
