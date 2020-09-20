import logging
from pathlib import Path
from sys import platform
import os
import joblib
from typing import List, Dict, Callable

import pandas as pd
import numpy as np
import yaml

from my_ml.utils.io import create_dir

log = logging.getLogger(__name__)


class DataCleaner:
    """Process the data.

    Parameters
    ----------
    data_path : str
        path/to/data
    save_dir : str
        path/to/save/dir
    random_state : int
        Random seed.

    Attributes
    ----------
    """

    def __init__(
        self, data_path: str, save_dir: str, random_state: int = 38,
    ):
        self._random_state = random_state
        self._config: Dict[str, int] = {**locals()}
        self._results_dir: Path = create_dir(Path(save_dir))
        # Dump config files.
        with open(self._results_dir / "config.yaml", "w") as steam:
            yaml.dump(self._config, steam)
        self._data_path = Path(data_path)

    def process_data(self):
        """Create the necessary data."""
        self._data: pd.DataFrame = self._load_data()
        self._data: pd.DataFrame = self._clean_features()
        self._test_train_split(test_prop=0.2, shuffle=True)
        save_path: Path = self._dump_data()
        if platform == "linux" or platform == "linux2":
            try:
                os.system("free -m")
            except:
                log.info("Tried printing memory information and could not.")
        log.info(f"Data dumped in {save_path}.")
        return save_path

    def _load_data(self) -> pd.DataFrame:
        """Load data in and make it square."""
        log.info("Loading data.")
        data = pd.read_csv(self._data_path)
        return data

    def _clean_features(self) -> pd.DataFrame:
        """Cleaning features."""
        data: pd.DataFrame = self._data
        log.info("Cleaning features.")
        # Columns that have a zero but should not. We can consider these
        # missing values.
        log.info(f"Finding missing data is labeled as 0: {data.describe().T}")
        nozero_cols: List = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
        ]
        data[nozero_cols]: pd.DataFrame = data[nozero_cols].replace(0, np.nan)
        # TODO: Consider moving this to a configuration file for easy
        # maintenance. This is unneccesarily nested in the event I'd like to
        # abstract this for ease of trying out new data cleaning methodologies
        # quickly.
        # TODO: Currently this is data leakage, but I just wanted to get it up
        # and running.
        imputing_funcs: Dict[str, Callable[[pd.Series], float]] = {
            "Glucose": np.nanmean,
            "BloodPressure": np.nanmean,
            "SkinThickness": np.nanmedian,
            "Insulin": np.nanmedian,
            "BMI": np.nanmedian,
        }
        for col in imputing_funcs.keys():
            func: Callable[[pd.Series], float] = imputing_funcs[col]
            data[col]: pd.Series = data[col].fillna(func(data[col]))
        log.info(f"Data has been imputed with mean/median: {data.describe().T}")
        return data

    def _test_train_split(self, test_prop: float, shuffle: bool = True):
        """Separate pandas dataframe into numpy arrays.

        Parameters
        ----------
        test_prop : float
            Proportion of the data to be allocated to the test set.
        shuffle : bool
            Whether to shuffle the data or not.
        """
        log.info("Separating the data for training and testing sets.")
        data: pd.DataFrame = self._data
        # Setting a seed if the random_set is not set to None.
        if self._random_state:
            np.random.seed(self._random_state)
        # Shuffling the dataframe.
        if shuffle:
            data: pd.DataFrame = data.sample(frac=1)
        y: np.ndarray = data["Outcome"].to_numpy()
        # Reshaping to be an m x 1 dimensional vector.
        y: np.ndarray = y.reshape(-1, 1)
        X: np.ndarray = data.drop(columns=["Outcome"]).to_numpy()
        # m is the number of examples.
        m = X.shape[0]
        # Randomly choosing indices for the test group.
        test_indices = np.random.choice([True, False], m, p=[test_prop, 1 - test_prop])
        self._X_train = X[~test_indices]
        self._X_test = X[test_indices]
        self._y_train = y[~test_indices]
        self._y_test = y[test_indices]

    def _dump_data(self) -> Path:
        """Dump the data."""
        log.info("Dumping the data to {self._results_dir}.")
        joblib.dump(self.__dict__, self._results_dir / "server.gz")
        return self._results_dir


if __name__ == "__main__":
    data_cleaner = DataCleaner(
        data_path="data", save_dir="output/clean", random_state=38
    )
    results_path = data_cleaner.process_data()
