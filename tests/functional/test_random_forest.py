from typing import Dict

import numpy as np
import joblib
import pytest

from my_ml.utils.scoring import get_f1


def test_random_forest_1():
    """Make sure no assertions are hit on the real data."""
    from my_ml.model.random_forest import RandomForest

    data: Dict = joblib.load("tests/data/server.gz")
    X_train: np.ndarray = data["_X_train"]
    y_train: np.ndarray = data["_y_train"]
    X_test: np.ndarray = data["_X_test"]
    random_forest = RandomForest(S=10)
    random_forest.fit(X_train, y_train)
    values: np.ndarray = random_forest.predict(X_test)


def test_random_forest_2():
    """Make sure we're in the ballpark."""
    # TODO: This was originally built based off a BRAF. This may fail as I
    # haven't tuned in the values here.
    from my_ml.model.random_forest import RandomForest

    data: Dict = joblib.load("tests/data/server.gz")
    X_train: np.ndarray = data["_X_train"]
    y_train: np.ndarray = data["_y_train"]
    X_test: np.ndarray = data["_X_test"]
    y_test: np.ndarray = data["_y_test"]
    random_forest = RandomForest(S=100)
    random_forest.fit(X_train, y_train)
    values: np.ndarray = random_forest.predict(X_test)
    # Making sure we are dealing with a two-class problem.
    assert values.shape[1] == 2
    y_preds: np.ndarray = (values > .5)[:, 1].astype(int)
    y_test: np.ndarray = y_test.reshape(-1)
    f1: float = get_f1(y_test, y_preds)
    # In the RandomForest paper, the authors claim to get a score of .66 +/- .07. I am
    # just making sure I'm in the ballpark.
    assert np.isclose(f1, .66, rtol=.1)
