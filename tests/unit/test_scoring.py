import numpy as np


def test_true_positive():
    """Test true positives function."""
    from my_ml.utils.scoring import true_positive

    y: np.ndarray = np.array([0, 1, 1, 1, 0, 1])
    y_preds: np.ndarray = np.array([0, 0, 1, 1, 1, 0])
    tp: float = true_positive(y, y_preds)
    assert np.isclose(tp, 2, rtol=.001)


def test_false_positive():
    """Test false positives function."""
    from my_ml.utils.scoring import false_positive

    y: np.ndarray = np.array([0, 0, 1, 1, 0, 1])
    y_preds: np.ndarray = np.array([1, 0, 1, 1, 1, 0])
    tp: float = false_positive(y, y_preds)
    assert np.isclose(tp, 2, rtol=.001)


def test_false_negative():
    """Test false negative function."""
    from my_ml.utils.scoring import false_negative

    y: np.ndarray = np.array([0, 0, 1, 1, 0, 1])
    y_preds: np.ndarray = np.array([1, 0, 0, 0, 1, 0])
    tp: float = false_negative(y, y_preds)
    assert np.isclose(tp, 3, rtol=.001)


def test_precision():
    """Test function."""
    from my_ml.utils.scoring import get_precision

    y: np.ndarray = np.array([0, 0, 1, 1, 1, 1])
    y_preds: np.ndarray = np.array([1, 0, 0, 0, 1, 0])
    prec: float = get_precision(y, y_preds)
    assert np.isclose(prec, .5, rtol=.001)


def test_recall():
    """Test function."""
    from my_ml.utils.scoring import get_recall

    y: np.ndarray = np.array([0, 0, 1, 1, 1, 1])
    y_preds: np.ndarray = np.array([1, 0, 0, 0, 1, 0])
    recall: float = get_recall(y, y_preds)
    assert np.isclose(recall, .25, rtol=.001)


def test_f1():
    """Test f1 function."""
    from my_ml.utils.scoring import get_f1

    y: np.ndarray = np.array([0, 0, 1, 1, 1, 1])
    y_preds: np.ndarray = np.array([1, 0, 0, 0, 1, 0])
    f1: float = get_f1(y, y_preds)
    assert np.isclose(f1, .3333333, rtol=.0000001)
