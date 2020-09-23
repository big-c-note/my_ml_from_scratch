from typing import Tuple

import numpy as np


def _split_data_slow(
    data: np.ndarray, threshold: np.ndarray, feature_j: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data on a given feature / threshold combo. Pulled from original
    code before adding Cython.
    """

    low_data: np.ndarray = data[np.where(data[:, feature_j] <= threshold)]
    high_data: np.ndarray = data[np.where(data[:, feature_j] > threshold)]
    low_y: np.ndarray = low_data[:, -1]
    high_y: np.ndarray = high_data[:, -1]
    return low_y, high_y, low_data[:, :-1], high_data[:, :-1]


def test_split_data_1():
    """Test _split_data() python function."""
    from my_ml.model._split_data_fast import _split_data

    for _ in range(100):
        for _ in range(100):
            n_features: int = np.random.randint(1, 10 + 1)
            # the top value in randint is ot included.
            feature_j: int = np.random.randint(0, n_features)
            for i, _ in enumerate(range(n_features)):
                if i == 0:
                    data: np.ndarray = np.random.rand(100).reshape(-1, 1) * 100
                else:
                    arr: np.ndarray = np.random.rand(100).reshape(-1, 1) * 100
                    data: np.ndarray = np.hstack((data, arr))
            val: int = np.random.randint(1, 100)
            low_y1, high_y1, low_X1, high_X1 = _split_data_slow(
                data, val, feature_j
            )
            low_y2, high_y2, low_X2, high_X2 = _split_data(
                data, val, feature_j
            )
            assert (low_y1 == low_y2).all() and (high_y1 == high_y2).all()
            assert (low_X1 == low_X2).all() and (high_X1 == high_X2).all()
