"""Split dataset."""
import json
from typing import List

from zenml.steps import BaseParameters, Output, step
from zenml.logger import get_logger

from ..src.voc_utils import get_annotations, create_train_test_split

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
def split_data(
    params: DatasetSplitParameters, jsonString: str
) -> Output(train=List[str], test=List[str]):
    """Split dataset into  train and test.

    It creates two labels text files `trainval.txt` and `test.txt`.

    Args:
        params (DatasetSplitParameters): Parameters for splitting dataset
        jsonString (str): string containing exported labels in json format

    Returns:
        train: a list of image files in the training dataset.
        test: a list of image files in the testing dataset.
    """
    # Convert string from json.dumps to json
    labelbox_export = json.loads(jsonString)

    # Get annotations
    annotations = get_annotations(
        params.image_base_dir, params.label_base_dir, labelbox_export
    )

    # Split dataset
    train, test = create_train_test_split(
        params.train_test_split_ratio, params.label_base_dir, annotations
    )

    return train, test
