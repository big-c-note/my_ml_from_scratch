"""
Usage: my_ml model [OPTIONS]

  Model the clean data.

Options:
  --data_path TEXT        Path to the the clean server object.
  --save_dir TEXT         Path to save the clean data.
  --random_state INTEGER  Integer to set a random seed.
  --help                  Show this message and exit.""
"""
import logging
from typing import List, Dict
from pathlib import Path

import click
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import trange

from my_ml.utils.io import create_dir
from my_ml.model.random_forest import RandomForest
from my_ml.utils.scoring import (
    true_positive,
    false_positive,
    get_recall,
    get_precision,
    get_f1,
)
from my_ml.utils.splits import val_train_split

log = logging.getLogger(__name__)


def _get_scoring_metrics(
    y: np.ndarray,
    y_preds: np.ndarray,
    results_dict: Dict,
    validation_set: str,
    threshold: float,
    kfolds: float,
) -> Dict:
    """
    Helper function to add the relevant scoring metrics.

    Parameters
    ----------
    y : np.ndarray
        The actual y values.
    y_preds : np.ndarray
        The predictions for y based on the feature data.
    results_dict : Dict
        A dictonary to store results in.
    threshold : float
        The threshold we are considering for the various metrics.
    kfolds : int
        The number of folds in the k fold cross validation.

    Returns
    -------
    results_dict : Dict
        Dictionary of scoring metrics.
    """
    fp: int = false_positive(y, y_preds)
    tp: int = true_positive(y, y_preds)
    recall: float = get_recall(y, y_preds)
    precision: float = get_precision(y, y_preds)
    f1: float = get_f1(y, y_preds)
    metrics: Dict[str, float] = {
        "false_pos": fp,
        "true_pos": tp,
        "recall": recall,
        "precision": precision,
        "fscore": f1,
    }
    # Add the scoring metrics to the results_dict object. We are calulating it
    # into an average over all kfolds.
    # On the test scoring, we do not need to average over k folds.
    if validation_set == "test":
        kfolds = 1
    for metric in metrics.keys():
        assert not np.isnan(metrics[metric])
        try:
            # Accumulating the average scores.
            results_dict[validation_set][metric][threshold] += metrics[metric] / kfolds
        except KeyError:
            # If there is no key, we need to initialize it.
            results_dict[validation_set][metric][threshold] = metrics[metric] / kfolds
    return results_dict


def _get_curves(results_dict: Dict, validation_set: str, save_dir: Path) -> Dict:
    """
    Helper function for getting AUROC, AUPRC, PRC and AUC.

    In retrospect, this helper function is a tad awkward. I might consider
    refactoring this into a few stand alone functions; auroc and auprc. I
    decided to calculate the AUROC and AUPRC in this way to save on iterations
    over thresholds, however it is a minor speed up.

    NOTE: AUROC and AUPRC are estimated from 10 data points using the
    trapizoidal rule for integration. A more accurate estimate would result as
    the number of points increases to inifity. Usually I would use an sklearn
    function for this, but I implemented it from scratch per the requirements.

    Parameters
    ----------
    results_dict : Dict
        Results dictionary.
    validation_set :str
        Either "cross_val" or "test". This tells us where to store the results.
    save_dir : Path
        Directory to save results to.
    """
    # Code to get a ROC graph and AUROC metric.
    tp_line: np.ndarray = _get_lines("true_pos", results_dict, validation_set)
    fp_line: np.ndarray = _get_lines("false_pos", results_dict, validation_set)
    plt.plot(fp_line, tp_line)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Graph: {validation_set}")
    plt.savefig(save_dir / f"roc_{validation_set}.png")
    plt.close()
    # This function is a coarse approximation of the area as the function is
    # continuous and I only computed 10 distinct intervals. Ordinarily I would
    # use sklearn for this metric.
    # Normalizing the data to be withing 0 and 1.
    # https://medium.com/@stallonejacob/data-science-scaling-of-data-in-python-ec7ad220b339
    tp_line_norm: np.ndarray = (tp_line - np.amin(tp_line)) / (
        np.amax(tp_line) - np.amin(tp_line)
    )
    fp_line_norm: np.ndarray = (fp_line - np.amin(fp_line)) / (
        np.amax(fp_line) - np.amin(fp_line)
    )
    # https://stackoverflow.com/questions/11973800/trapz-giving-weird-results
    auroc: float = np.trapz(tp_line_norm, -fp_line_norm)
    results_dict[validation_set]["auroc"] = auroc
    # Code to get PRC graph and AUPRC metric.
    recall_line: np.ndarray = _get_lines("recall", results_dict, validation_set)
    precision_line: np.ndarray = _get_lines("precision", results_dict, validation_set)
    plt.plot(recall_line, precision_line)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PRC Graph: {validation_set}")
    plt.savefig(save_dir / f"prc_{validation_set}.png")
    plt.close()
    # This function is a coarse approximation of the area as the function is
    # continuous and I only computed 10 distinct intervals. Ordinarily I would
    # use sklearn for this metric.
    auprc: float = np.trapz(precision_line, -recall_line)
    results_dict[validation_set]["auprc"] = auprc
    return results_dict


def _get_lines(metric: str, results_dict: Dict, validation_set) -> np.ndarray:
    """Get the arrays iof metrics associated with different thesholds."""
    metric_values: List[float] = np.array(
        list(results_dict[validation_set][metric].values())
    )
    return metric_values


@click.command()
@click.option(
    "--data_path",
    default="tests/data/server.gz",
    help="Path to the the clean server object.",
)
@click.option("--save_dir", default="output/model", help="Path to save the clean data.")
@click.option("--random_state", default=None, help=("Not Implemented"))
def model_data(data_path: str, save_dir: str, random_state: int = 38):
    """Model the clean data."""
    log.info("Loading the clean data.")
    data: Dict = joblib.load(Path(data_path))
    X: np.ndarray = data["_X_train"]
    y: np.ndarray = data["_y_train"]

    # Number of folds in the k-fold cross validation.
    kfolds: int = 10
    # The model parameters to iterate over and try.
    # TODO: A tad overkill of an object for one parameter, but I refactored
    # from a model with many parameters.
    model_params: List[Dict[str, float]] = [{"S": 100}]
    # Iterate over model parameters.
    # There is only one but you could add more!
    for params in model_params:
        results_dict: Dict = {
            "model": None,
            "cross_val": {
                "false_pos": {},
                "true_pos": {},
                "recall": {},
                "precision": {},
                "fscore": {},
            },
            "test": {
                "false_pos": {},
                "true_pos": {},
                "recall": {},
                "precision": {},
                "fscore": {},
            },
        }
        # Iterate over the number of k folds.
        log.info(f"""Starting {kfolds} Fold Cross Validation""")
        for cval_i in trange(kfolds):
            log.info(f"On fold {cval_i} of {kfolds} for cross validation.")
            # Get the cross validation data for this iteration.
            X_train, y_train, X_val, y_val = val_train_split(X, y, cval_i, kfolds)
            random_forest: RandomForest = RandomForest(S=params["S"])
            random_forest.fit(X_train, y_train)
            predictions: np.ndarray = random_forest.predict(X_val)
            assert predictions.shape[0] == y_val.shape[0]
            validation_set: str = "cross_val"
            # The thresholds to calulate scoring metrics for. Should result in
            # increments of .1.
            num_thresholds: int = 11
            try:
                assert num_thresholds == 11
            except AssertionError:
                raise NotImplementedError("`num_thresholds` must be set to 11.")
            thresholds: np.ndarray = np.around(
                np.linspace(0, 1, num_thresholds), decimals=1
            )
            for threshold in thresholds:
                # Slightly awkward syntax, but need to turn the m x 2
                # dimensional probability vectora down to (m,).
                y_preds: np.ndarray = (predictions > threshold)[:, 1].astype(
                    int
                ).reshape(-1)
                y_val: np.ndarray = y_val.reshape(-1)
                results_dict: Dict = _get_scoring_metrics(
                    y_val, y_preds, results_dict, validation_set, threshold, kfolds
                )
        # Creating a path to save the data to.
        save_path: Path = create_dir(Path(save_dir))
        # Helper function for calculating AUROC, AUPRC, ROC and PRC
        results_dict: Dict = _get_curves(results_dict, "cross_val", save_path)
        # Now calculating metrics for the test group. I'm reloading the data in
        # case anything was altered.
        log.info("Getting metrics for the test data.")
        X: np.ndarray = data["_X_train"]
        y: np.ndarray = data["_y_train"]
        X_test: np.ndarray = data["_X_test"]
        y_test: np.ndarray = data["_y_test"]
        # No longer need a column vector.
        y_test: np.ndarray = y_test.reshape(-1)
        # Need to refit the model based on all training data. We can save this
        # one. All we really care about are the parameters used.
        random_forest: RandomForest = RandomForest(S=params["S"])
        random_forest.fit(X, y)
        predictions: np.ndarray = random_forest.predict(X_test)
        validation_set: str = "test"
        num_thresholds: int = 11
        thresholds: np.ndarray = np.around(
            np.linspace(0, 1, num_thresholds), decimals=1
        )
        for threshold in thresholds:
            # Slightly awkward syntax, but need to turn the m x 2
            # dimensional probability vectora down to (m,).
            y_preds: np.ndarray = (predictions > threshold)[:, 1].astype(int).reshape(
                -1
            )
            results_dict: Dict = _get_scoring_metrics(
                y_test, y_preds, results_dict, validation_set, threshold, kfolds
            )
        results_dict: Dict = _get_curves(results_dict, "test", save_path)
        # Saving the model.
        results_dict["model"] = random_forest
        log.info(f"Dumping the results and model configs to {save_path}.")
        joblib.dump(results_dict, save_path / "results.gz")


if __name__ == "__main__":
    model_data()
