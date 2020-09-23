from typing import Optional, Tuple, Dict

import numpy as np

from my_ml.model._split_data_fast import _split_data


class Node:
    def __init__(
        self,
        threshold: float = None,
        feature_j: int = None,
        leaf: bool = True,
        value: Optional[int] = None,
    ):
        """
        Nodes contain information to guide the navigation of the decision
        tree.

        Parameters
        ----------
        threshold : float
            Value to split on. This value was determined to split into the
            groups with the lowest gini score.
        feature_j : int
            Index of the jth feature.
        leaf : bool
            Whether or not the Node is a leaf.
        value : Optional[int]
            The class with the highest number of members at a leaf node.
        """
        self.threshold = threshold
        self.feature_j = feature_j
        self.is_leaf = leaf
        self.value = value

    def set_left(self, left: "Node"):
        """Set the left child."""
        assert isinstance(left, Node)
        self._left = left

    def set_right(self, right: "Node"):
        """Set the right child."""
        assert isinstance(right, Node)
        self._right = right

    def get_left(self) -> "Node":
        """Get the left child."""
        assert self._left
        return self._left

    def get_right(self) -> "Node":
        """Get the right child."""
        assert self._right
        return self._right


class DecisionTree:
    def __init__(
        self, random_features: bool = False,
    ):
        """
        The DecisionTree class contains methods that string together a binary
        tree. The nodes are given features and threshold values to split the
        data to minimizes the impurity of the nodes.

        Parameters
        ----------
        random_features : bool
            If False, we iterate over all features to find the best feature /
            threshold combo that minimizes the impurity of spitting the data.

            If True, we randomly generate ceil(sqrt(num_features)) features at
            each level to try. This is necesseary for random forests, as
            shortening the dimentionality of the trees reduces variance.

        Methods
        -------
        fit(X: np.ndarray, y: np.ndarray)
            Method for building a decision tree on the given data. X is an m x
            n dimensional array, and y is an m x 1 dimensional array.
        predict(X)
            Method for traversing the tree with an m x n dimensional array of
            examples.
        """
        self._root: Optional[Node] = None
        self._K: Optional[int] = None
        self._num_features: Optional[int] = None
        self._random_features: bool = random_features
        self._num_random_features: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Method for building a decision tree on the given data.

        X : np.ndarray
            X is an m x n dimensional array of features. m is the number of
            examples and n is the number of features.
        y : np.ndarray
            y is an m x 1 dimensional array of labels. This should take on
            labels in range(k).
        """
        # TODO: It would be faster to accept ints. It would take checking the
        # dtype and using a separate split_data function that used ints.
        try:
            assert X.dtype == float
        except AssertionError:
            raise NotImplementedError("X must be a float dtype.")
        self._root = self._build_tree(X, y, depth=0)

    def predict_value(self, x: np.array, subtree=None) -> int:
        """Predict a single value given a single set of features."""
        # Set the binary_tree.
        if not subtree:
            assert self._root
            binary_tree: Node = self._root
        else:
            binary_tree: Node = subtree
        value: float = x[binary_tree.feature_j]
        # Get the branch based on the optimal splitting threshold.
        assert binary_tree.threshold
        if value <= binary_tree.threshold:
            child: Node = binary_tree.get_left()
        else:
            child: Node = binary_tree.get_right()
        assert child
        if child.is_leaf:
            # We found a leaf! Return the majority class (should be pure).
            return child.value
        return self.predict_value(x, child)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict m values given an m x n dimentional feature array."""
        predictions = np.array([self.predict_value(instance) for instance in X])
        return predictions

    def _build_tree(self, X: np.ndarray, y: np.array, depth: int) -> Node:
        """
        Method for building a decision tree on the given data.
        X : np.ndarray
            X is an m x n dimensional array of features. m is the number of
            examples and n is the number of features.
        y : np.ndarray
            y is an m x 1 dimensional array of labels. This should take on
            labels in range(k).
        depth : int
            Not used, but can be used for determining a max_depth cutoff. If
            the data contained more examples, we may consider implementing
            this as it can reduce the training time.
        """
        # TODO: There can be a speed-up if the training data contains many
        # examples. I can implement a max_depth that would truncate the tree at
        # that depth, returning the majority class value. This is not
        # neccessary for our tiny Pima dataset.
        if not self._K:
            # This is the number of labels.
            self._K = len(set(y.flat))
        if not self._num_features:
            self._num_features = X.shape[1]
        set_num_random_featues = self._random_features and not self._num_random_features
        if set_num_random_featues:
            # https://www.youtube.com/watch?v=4EOCQJgqAOY 9:30
            # The number of random features has a pre-determined near optimal
            # value.
            self._num_random_features: int = int(np.ceil(np.sqrt(self._K)))
        # I'm only supporting numeric data.
        assert np.issubdtype(X.dtype, np.number)
        assert np.issubdtype(y.dtype, np.number)
        # We should never experience fewer features than how many we start with.
        assert X.shape[1] == self._num_features
        # X and y should always have the same number of examples.
        assert X.shape[0] == y.shape[0]
        if len(set(y.flat)) == 1:
            # We have found a pure leaf. Return a leaf node with a prediction
            # value.
            y = y.astype(int)
            node = Node()
            node.is_leaf = True
            node.value = np.bincount(y).argmax()
            return node

        best_gini_score: float = np.inf
        best_feature_j: Optional[int] = None
        best_threshold: Optional[float] = None

        # Decide on random features if self._random_features.
        all_features: np.ndarray = np.array(range(X.shape[1]))
        if self._random_features:
            features: np.ndarray = np.random.choice(
                all_features, size=self._num_random_features, replace=False
            )
        else:
            features: np.ndarray = all_features

        # If all of the features we want to try are the same, return a leaf
        # node with the most common value of y as a prediction. This is our
        # other base case.
        if (X[:, features] == X[:, features][0]).all():
            y = y.astype(int)
            node = Node()
            node.is_leaf = True
            node.value = np.bincount(y).argmax()
            return node

        for feature_j in features:
            # Making this a m x 1 dimensional vector.
            y: np.ndarray = y.reshape(-1, 1)
            # Sorting the numeric values in X to find possible thresholds.
            x: np.ndarray = X[:, feature_j].reshape(-1, 1)
            x_sort: np.ndarray = np.unique(x[x[:, 0].argsort()], axis=0)
            # Find the mid points to try.
            if len(set(x_sort.flat)) == 1:
                # If there is only one value, this means we only have one x
                # value, we can't split on one value. Move to the next feature
                # to try.
                continue
            else:
                thresholds = (x_sort[1:, 0] + x_sort[:-1, 0]) / 2
            # Adding y to the end of the feature vector. It is now a m x n+1
            # sized matrix.
            data: np.ndarray = np.hstack((X, y))
            gini_score, threshold, data_dict = self._find_best_gini_score(
                data, thresholds, feature_j
            )
            # Stack the data.
            if gini_score < best_gini_score:
                best_gini_score = gini_score
                best_feature_j = feature_j
                best_threshold = threshold
                left_y: np.ndarray = data_dict["left_y"]
                left_X: np.ndarray = data_dict["left_X"]
                right_y: np.ndarray = data_dict["right_y"]
                right_X: np.ndarray = data_dict["right_X"]

        node = Node(threshold=best_threshold, feature_j=best_feature_j)
        node.is_leaf = False
        # Set the children.
        node.set_left(self._build_tree(left_X, left_y, depth + 1))
        node.set_right(self._build_tree(right_X, right_y, depth + 1))
        return node

    def _find_best_gini_score(
        self, data: np.ndarray, thresholds: np.ndarray, feature_j: int
    ) -> Tuple[float, float, Dict[str, np.ndarray]]:
        """
        Iterate over thresholds and find the best split.
        """
        tmp_best_gini_score: float = np.inf
        tmp_best_threshold: float
        # TODO: I am imagining a log(n) speedup where by I check the middle
        # threshold first, and then the mid point to both ends. I then
        # recursively do this, going in the direction of gini score that is
        # lower. The idea is that gini score is a concave function so you don't
        # have to iterate through all values.
        for threshold in thresholds:
            # Split the data on the threshold.
            low_y, high_y, low_X, high_X = _split_data(
                data, threshold, feature_j
            )
            assert isinstance(self._K, int)
            gini_score: float = self._get_gini_score(low_y, high_y, self._K)
            if gini_score < tmp_best_gini_score:
                tmp_best_gini_score = gini_score
                tmp_best_threshold = threshold
                data_dict: Dict[str, np.ndarray] = {
                    "left_y": low_y,
                    "left_X": low_X,
                    "right_y": high_y,
                    "right_X": high_X,
                }
        return tmp_best_gini_score, tmp_best_threshold, data_dict

    @staticmethod
    def _get_gini_score(low_y: np.ndarray, high_y: np.ndarray, K: int) -> float:
        """
        Find the gini score of this potential split.
        """
        low_gini: float = 1.0
        high_gini: float = 1.0
        assert isinstance(K, int)
        for k in range(K):
            low_k: np.ndarray = low_y[low_y == k]
            high_k: np.ndarray = high_y[high_y == k]
            # If low_y is zero, we've found a pure split. TODO: Consider
            # changing the logic to return here or sooner for faster
            # computation.
            try:
                low_prop: float = (low_k.shape[0] / low_y.shape[0])
            except ZeroDivisionError:
                low_prop: float = 0
            try:
                high_prop: float = (high_k.shape[0] / high_y.shape[0])
            except ZeroDivisionError:
                high_prop: float = 0
            low_gini -= low_prop ** 2
            high_gini -= high_prop ** 2
        total_data_size: int = low_y.shape[0] + high_y.shape[0]
        low_prop: float = low_y.shape[0] / total_data_size
        high_prop: float = high_y.shape[0] / total_data_size
        gini_score: float = (low_prop * low_gini) + (high_prop * high_gini)
        # Gini score ranges between 0 and 1.
        assert gini_score >= 0 and gini_score <= 1
        return gini_score
