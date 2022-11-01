"""Split dataset."""
import json

import mlflow
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

from ..src.voc_utils import create_train_test_split, get_annotations

logger = get_logger(__name__)


class DatasetSplitParameters(BaseParameters):
    """Dataset parameters."""

    # Path to image directory containing images
    image_base_dir: str

    # Path to label directory that will be created
    label_base_dir: str

    # Split ratio to split dataset into trainval and test dataset
    train_test_split_ratio: float


@step
def split_data(params: DatasetSplitParameters, jsonString: str) -> Output():
    """Split dataset into  train and test.

    It creates two labels text files `trainval.txt` and `test.txt`.

    Args:
        params (DatasetSplitParameters): Parameters for splitting dataset
        jsonString (str): string containing exported labels in json format
    """
    # Convert string from json.dumps to json
    labelbox_export = json.loads(jsonString)

    # Get annotations
    annotations = get_annotations(
        params.image_base_dir, params.label_base_dir, labelbox_export
    )

    # Split dataset
    create_train_test_split(
        params.train_test_split_ratio, params.label_base_dir, annotations
    )

    # log split ratio to mlflow
    mlflow.log_param("train_test_split_ratio", params.train_test_split_ratio)
