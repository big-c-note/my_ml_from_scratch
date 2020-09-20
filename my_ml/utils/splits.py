from typing import List

import numpy as np


def val_train_split(X: np.ndarray, y: np.ndarray, cval_i: int, kfolds: int):
    """
    Create a cross validation split based on training data.

    Parameters
    ----------
    X : np.ndarray
        m x n feature matrix.
    y : np.ndarray
        m x 1 or (m,) outcome vector.
    cval_i : int
        range(0, kfolds-1). This will determine what indices we use to
        partition the data into k folds.
    kfolds : int
        Number of folds.

    Returns
    -------
    X_train : np.ndarray
        All training data minus the indices as determined by cval_i.
    y_train : np.ndarray
        All the outcome training data minus the indices as determined by
        cval_i.
    X_val : np.ndarray
        A partition of X as determined by cval_i.
    y_val : np.ndarray
        A partition of y as determined by cval_i.
    """
    # Find the indices associated with splits that would yeild kfolds.
    indices: np.ndarray = np.round(np.linspace(0, X.shape[0] + 1, kfolds + 1)).astype(int)
    start_index: int = indices[cval_i]
    end_index: int = indices[cval_i + 1]
    # Making this a column vector.
    y: np.ndarray = y.reshape(-1, 1)
    # Partitioning the validation data.
    X_val: np.ndarray = X[start_index:end_index, :]
    y_val: np.ndarray = y[start_index:end_index, :]
    # Get the indices to remove.
    # At the end of the indices we cannot over shoot or we will get an out of
    # bounds error.
    if end_index - 1 == X.shape[0]:
        rm_indices: List[int] = list(range(start_index, end_index - 1))
    else:
        rm_indices: List[int] = list(range(start_index, end_index))
    try:
        X_train: np.ndarray = np.delete(X, rm_indices, axis=0)
    except:
        import ipdb; ipdb.set_trace()
    y_train: np.ndarray = np.delete(y, rm_indices, axis=0)
    return X_train, y_train, X_val, y_val
