from typing import List, Optional, Union
from multiprocessing import cpu_count
import logging

import numpy as np
import joblib
from joblib import Parallel, delayed
from tqdm import trange

from my_ml.model.decision_tree import DecisionTree

log = logging.getLogger(__name__)


class RandomForest:
    def __init__(self, S: Union[int, float]):
        """
        Base class for the Random forest. This class implements the import
        build forest method as well as the predict method.

        Parameters
        ----------
        S : int
            This is the number of trees for the random forest.

        Methods
        -------
        predict(X: np.ndarray)
            Returns predictions for a given feature set. The predictions come
            out as a probability matrix that is m x k in dimensionality. That
            is, the 0th column is each examples probability of being 0 and the
            1st column is the each examples probability of being 1.
        """
        self._S: Union[int, float] = S
        self._forest: List = []
        self._num_features: Optional[int] = None
        self._num_training_examples: Optional[int] = None
        # This is used for multi-processing. This allows for a speed up of (#
        # of cores)X
        self._cores: int = cpu_count()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Method for fitting feature X (m x n) to y (m x 1) or (m,)."""
        log.info("Creating decision trees.")
        if not self._num_features:
            self._num_features = X.shape[1]
        self._build_forest(X, y)
        assert len(self._forest) == self._S

    def predict(self, X: np.ndarray, probability: bool = True):
        """
        Return matrix of probabilities. 0th column references probability of
        being class 0 and the 1st column references the probability of being
        class 1.
        """
        assert self._forest
        assert len(self._forest) == self._S
        try:
            assert X.shape[0] > 1 and X.shape[1] > 1
        except AssertionError:
            raise NotImplementedError(
                """
                I'm not supporting predictions of single examples. It's easy to
                implement this functionality if yo uneed it.
                """
            )
        predictions: List = []
        # Iterate through the forest and create a prediction for each tree.
        for dtree in self._forest:
            predictions.append(dtree.predict(X))
        # Transposing the matrix for ease. This may not be needed.
        predictions: np.ndarray = np.array(predictions).T
        # Probabilities for the majority class.
        prob_maj: np.ndarray = np.count_nonzero(
            predictions == 0, axis=1
        ) / predictions.shape[1]
        # Probabilities for the minority class.
        prob_min: np.ndarray = np.count_nonzero(
            predictions == 1, axis=1
        ) / predictions.shape[1]
        # Making these column vectors.
        prob_maj: np.ndarray = prob_maj.reshape(-1, 1)
        prob_min: np.ndarray = prob_min.reshape(-1, 1)
        # Stacking these column vectors side by side.
        predictions: np.ndarray = np.hstack((prob_maj, prob_min))
        # TODO: Need to add noise in the event that there is a tie. With
        # sufficeintly large values or odd values of parameter S, this should
        # not be a problem. However, all equal probability class predictions
        # will resolve to the same prediction (should be random).
        return predictions

    def _build_forest(self, X: np.ndarray, y: np.ndarray, p: float = 1):
        """Build a forest of decision trees."""
        # Making y a column vector.
        y: np.ndarray = y.reshape(-1, 1)
        # Adding the y outcome vector to the end of the feature vector to create
        # T.
        T: np.ndarray = np.hstack((X, y))
        # Checking dimensions. This will run on T and Tc which will have a
        # different number of examples, most likely.
        assert self._num_features
        assert T.shape[1] == self._num_features + 1
        # Get the number of trees from S * p. p will be (1 - p) for T and p for
        # Tc.
        num_trees: int = int(np.round(self._S * p))
        # Checking these sum to S.
        assert np.round(self._S * p) + np.round(self._S * (1 - p)) == self._S
        # Adding multi-processing for an easy speedup. Adding progress bar for
        # viewing how long the job will take.
        log.info(
            """
            Showing progress on one of two forests to be created. Distributing
            compute over all available cores.
            """
        )
        forest: List = Parallel(n_jobs=self._cores)(
            delayed(self._build_forest_helper)(T) for i in trange(num_trees)
        )
        # Add forest to our list of decision trees.
        self._forest += forest

    @staticmethod
    def _build_forest_helper(T) -> DecisionTree:
        # Bagging the T dataset. Essentially I am sampling with replacement and
        # creating T_rand which has the same dimensions as T.
        T_rand: np.ndarray = T[np.random.randint(T.shape[0], size=T.shape[0]), :]
        assert T_rand.shape == T.shape
        # Separating X and y.
        X_rand: np.ndarray = T_rand[:, :-1]
        y_rand: np.ndarray = T_rand[:, -1]
        # random_features=True means we will randomly generate a subset of
        # features to use in each decision tree. This helps reduce
        # variance.
        dtree = DecisionTree(random_features=True)
        dtree.fit(X_rand, y_rand)
        # Add our decision tree to the forest.
        return dtree


if __name__ == "__main__":
    cat = joblib.load("tests/data/server.gz")
    X_train: np.ndarray = cat["_X_train"]
    X_test: np.ndarray = cat["_X_test"]
    y_train: np.ndarray = cat["_y_train"]
    y_test: np.ndarray = cat["_y_test"]
    braf = BRAF(k=10, p=0.5, S=500)
    braf.fit(X_train, y_train)
    values: np.ndarray = braf.predict(X_test)
    import ipdb

    ipdb.set_trace()
