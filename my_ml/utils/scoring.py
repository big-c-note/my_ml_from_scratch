"""
Written from scratch with these guides as reference.
https://en.wikipedia.org/wiki/Precision_and_recall
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for\
        -classification-in-python/
"""
import numpy as np


def true_positive(y: np.ndarray, y_preds: np.ndarray) -> int:
    """Find the number of true positives."""
    assert y.shape == y_preds.shape
    return y[y_preds == 1].sum()


def false_positive(y: np.ndarray, y_preds: np.ndarray) -> int:
    """Find the number of false positives."""
    assert y.shape == y_preds.shape
    return y_preds[y == 0].sum()


def false_negative(y: np.ndarray, y_preds: np.ndarray) -> int:
    """Find the number of false negatives."""
    assert y.shape == y_preds.shape
    return y[y_preds == 0].sum()


def get_precision(y: np.ndarray, y_preds: np.ndarray) -> float:
    """Return precision."""
    assert y.shape == y_preds.shape
    tp: int = true_positive(y, y_preds)
    fp: int = false_positive(y, y_preds)
    # If we don't make any positive predictions, then we don't get any wrong.
    if tp == 0 and fp == 0:
        precision: float = 1.0
    else:
        precision: float = tp / (tp + fp)
    assert not np.isnan(precision)
    return precision


def get_recall(y: np.ndarray, y_preds: np.ndarray) -> float:
    """Return recall."""
    assert y.shape == y_preds.shape
    tp: int = true_positive(y, y_preds)
    fn: int = false_negative(y, y_preds)
    recall: float = tp / (tp + fn)
    assert not np.isnan(recall)
    return recall


def get_f1(y: np.ndarray, y_preds: np.ndarray) -> float:
    """Return f1 score."""
    assert y.shape == y_preds.shape
    recall: float = get_recall(y, y_preds)
    precision: float = get_precision(y, y_preds)
    # If either is zero, then out f score is 0.
    # https://en.wikipedia.org/wiki/F1_score
    if precision == 0 or recall == 0:
        f1: float = 0
    else:
        f1: float = (2 * precision * recall) / (precision + recall)
    assert not np.isnan(f1)
    return f1
