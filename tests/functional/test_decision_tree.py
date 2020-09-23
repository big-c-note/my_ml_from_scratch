from typing import Dict

import numpy as np
import joblib


def test_decision_tree_1():
    """Small repex to alert me to any issue."""
    from my_ml.model.decision_tree import DecisionTree

    X: np.ndarray = np.array(
        [[15, 0, 0], [12, 1, 0], [4, 1, 0], [8, 0, 1], [0, 0, 0]], dtype=float
    )
    y: np.ndarray = np.array([0, 0, 1, 1, 1], dtype=float).reshape(-1, 1)
    dtree = DecisionTree()
    dtree.fit(X, y)
    X_test: np.ndarray = np.array([[0, 1, 0], [1, 0, 0]])
    predictions: np.ndarray = dtree.predict(X_test)
    assert np.all(predictions == 1)


def test_decision_tree_2():
    """Make sure no assertions are hit on the real data."""
    from my_ml.model.decision_tree import DecisionTree

    data: Dict = joblib.load("tests/data/server.gz")
    X: np.ndarray = data["_X_train"]
    y: np.ndarray = data["_y_train"]
    dtree = DecisionTree()
    dtree.fit(X, y)
    X_test: np.ndarray = data["_X_test"]
    predictions: np.ndarray = dtree.predict(X_test)


def test_decision_tree_3():
    """
    Make sure there aer no issues when using the random features
    parameter.
    """
    from my_ml.model.decision_tree import DecisionTree

    data: Dict = joblib.load("tests/data/server.gz")
    X: np.ndarray = data["_X_train"]
    y: np.ndarray = data["_y_train"]
    for i in range(10):
        dtree = DecisionTree()
        dtree.fit(X, y)
        X_test: np.ndarray = data["_X_test"]
        predictions: np.ndarray = dtree.predict(X_test)
