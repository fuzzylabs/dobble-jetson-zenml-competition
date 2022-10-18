"""Run all pipelines."""

import click
from rich import print

from steps.ingest_data import ingest_data
from steps.labelbox_to_voc import prepare_labels_step

# from steps.split_data import split_data
# from steps.validate_data import validate_data
# from steps.create_data_release import create_data_release

from pipelines.data_pipeline import data_pipeline


def run_data_pipeline():
    """Run all steps in data pipeline."""
    pipeline = data_pipeline(ingest_data(), prepare_labels_step())
    pipeline.run(config_path="config_data_pipeline.yaml")


@click.command()
@click.option(
    "--use_data_pipeline", "-dp", is_flag=True, help="Run data pipeline"
)
def main(use_data_pipeline: bool):
    """Run all pipelines.

    Args:
        use_data_pipeline (bool): enable running data pipeline

    Examples::
        python run.py -dp
    """
    if use_data_pipeline:
        print("Running data pipeline")
        run_data_pipeline()


if __name__ == "__main__":
    main()
