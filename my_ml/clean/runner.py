"""
Usage: my_ml clean [OPTIONS]

  Clean the raw data.

Options:
  --data_path TEXT        Path to the raw data.
  --save_dir TEXT         Path to save the clean data.
  --random_state INTEGER  Integer to set a random seed.
  --help                  Show this message and exit.""
"""
import logging

import click

from my_ml.clean.data_cleaner import DataCleaner

log = logging.getLogger(__name__)


@click.command()
@click.option("--data_path", default="data/diabetes.csv", help="Path to the raw data.")
@click.option("--save_dir", default="output/clean", help="Path to save the clean data.")
@click.option("--random_state", default=38, help=("Integer to set a random seed."))
def clean_data(data_path: str, save_dir: str, random_state: int):
    """Clean the raw data."""
    log.warning(
        "You have a random state set. This is useful for reproducing results."
    )
    data_cleaner = DataCleaner(
        data_path=data_path,
        save_dir=save_dir,
        random_state=random_state,
    )
    save_path = data_cleaner.process_data()


if __name__ == "__main__":
    clean_data()
