"""Fetch model from MLFlow server."""
import os

import mlflow
import onnx


def print_artifact_info(artifact):
    """Utility to print artifact information."""
    print("artifact: {}".format(artifact.path))
    print("is_dir: {}".format(artifact.is_dir))
    print("size: {}".format(artifact.file_size))


def get_models_from_experiment_tracker(
    tracking_uri: str, run_id: str, out_dir: str
):
    """Fetch model from mlflow server.

    Args:
        tracking_uri (str): Tracking uri for MLFlow server
        run_id (str): Run ID for MLflow experiment
        out_dir (str): Output directory path to save onnx model

    Raises:
        e: If `run_id` is invalid or not found in MLflow server
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    if run_id != "":
        print(
            f"Choosing MLFlow Experiment with run id : {run_id} for deployment"
        )
        try:
            run_data_dict = client.get_run(run_id).data.to_dictionary()
            print(run_data_dict)
        except Exception as e:
            print(
                f"Run id: {run_id} seems to be incorrect. Please check it again against MLFlow dashboard."
            )
            raise e

        # list artifacts
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            print_artifact_info(artifact)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # download onnx model
        client.download_artifacts(run_id, "onnx_model", out_dir)


def load_model(onnx_path: str):
    """Load and check onnx model.

    Args:
        onnx_path (str): Path to onnx model (must end in `.onnx`)
    """
    print("Checking ONNX model...")
    onnx_model = onnx.load(onnx_path)
    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
    else:
        print("The model is valid!")


def fetch_onnx_from_mlflow(tracking_uri: str, run_id: str, out_dir: str):
    """Fetch and load onnx model.

    Args:
        tracking_uri (str): Tracking uri for MLFlow server
        run_id (str): Run ID for MLflow experiment
        out_dir (str): Output directory path to save onnx model

    Raises:
        FileNotFoundError: If onnx model is not found at `{out_dir}/onnx_model/model.onnx` path
    """
    get_models_from_experiment_tracker(tracking_uri, run_id, out_dir)
    onnx_path = f"{out_dir}/onnx_model/model.onnx"
    if os.path.isfile(onnx_path):
        load_model(onnx_path)
    else:
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")


# if __name__ == "__main__":
#     tracking_uri = "http://localhost:5000"
#     run_id = "d852ad88f6544b379f6e2d36b1436218"
#     out_dir = "model"
#     fetch_onnx_from_mlflow(tracking_uri, run_id, out_dir)
