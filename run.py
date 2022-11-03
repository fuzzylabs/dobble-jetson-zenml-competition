"""Run all pipelines."""
import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from pipelines.data_pipeline.data_pipeline import data_pipeline
from pipelines.training_pipeline.training_pipeline import training_pipeline
from steps.create_data_loader.create_data_loader_step import create_data_loader
from steps.download_data.download_data_step import download_data
from steps.evaluate_model.evaluate_model_step import evaluate_model
from steps.export_onnx.export_onnx_step import export_onnx
from steps.ingest_data.ingest_data_step import ingest_data
from steps.labelbox_to_voc.labelbox_to_voc_step import prepare_labels_step
from steps.split_data.split_data_step import split_data
from steps.trainer.trainer_step import trainer
from steps.upload_data.upload_data_step import upload_data
from steps.validate_data.validate_data_step import data_integrity_check
from steps.validate_data_model.validate_data_model_step import (
    validate_data_model,
)


def run_data_pipeline():
    """Run all steps in data pipeline."""
    pipeline = data_pipeline(
        ingest_data(), prepare_labels_step(), split_data(), upload_data()
    )
    pipeline.run(
        config_path="pipelines/data_pipeline/config_data_pipeline.yaml"
    )


def run_training_pipeline():
    """Run all steps in training pipeline."""
    pipeline = training_pipeline(
        download_data(),
        create_data_loader(),
        data_integrity_check(),
        trainer(),
        validate_data_model(),
        evaluate_model(),
        export_onnx(),
    )
    pipeline.run(
        config_path="pipelines/training_pipeline/config_training_pipeline.yaml"
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

    # Run all steps in training pipeline
    if use_train_pipeline:
        print("Running training pipeline")
        run_training_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
    )


if __name__ == "__main__":
    main()
