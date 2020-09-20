import click

from my_ml.model.runner import model_data
from my_ml.clean.runner import clean_data


@click.group()
def cli():
    """The CLI for the my_ml package that groups the various scripts.
    """
    pass


cli.add_command(model_data, name="model")
cli.add_command(clean_data, name="clean")
