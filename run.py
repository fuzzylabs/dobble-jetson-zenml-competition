"""Run all pipelines."""
import click
from rich import print

from steps.ingest_data.ingest_data_step import ingest_data
from steps.labelbox_to_voc.labelbox_to_voc_step import prepare_labels_step
from steps.split_data.split_data_step import split_data

# from steps.validate_data.validate_data_step import validate_data
# from steps.create_data_release.create_data_release_step import create_data_release

from pipelines.data_pipeline.data_pipeline import data_pipeline

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


def run_data_pipeline():
    """Run all steps in data pipeline."""
    pipeline = data_pipeline(ingest_data(), prepare_labels_step(), split_data())
    pipeline.run(
        config_path="pipelines/data_pipeline/config_data_pipeline.yaml"
    )


@click.command()
@click.option(
    "--use_data_pipeline", "-dp", is_flag=True, help="Run data pipeline"
)
@click.option(
    "--use_train_pipeline", "-tp", is_flag=True, help="Run training pipeline"
)
def main(use_data_pipeline: bool, use_train_pipeline: bool):
    """Run all pipelines.

    Args:
        use_data_pipeline (bool): enable running data pipeline
        use_train_pipeline (bool): enable running training pipeline

    Examples:
        python run.py -dp      # run data pipeline only
        python run.py -tp      # run training pipeline only
        python run.py -dp -tp  # run both data and training pipeline
    """
    # Run all steps in data pipeline
    if use_data_pipeline:
        print("Running data pipeline")
        run_data_pipeline()
        print(
            "Now run \n "
            f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
            "To inspect your experiment runs within the mlflow UI.\n"
        )
    # Run all steps in training pipeline
    if use_train_pipeline:
        pass


if __name__ == "__main__":
    main()
