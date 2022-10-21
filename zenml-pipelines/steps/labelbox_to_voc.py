"""Convert labels from labelbox format to VOC format."""
import json

from zenml.steps import BaseParameters, Output, step
from zenml.logger import get_logger

from .src.voc_utils import (
    create_data_directories,
    get_annotations,
    get_labels,
    save_annotations_to_xml,
    save_labels,
)

logger = get_logger(__name__)


class DatasetParameters(BaseParameters):
    """Dataset parameters."""

    # Path to image directory containing images
    image_base_dir: str

    # Path to label directory that will be created
    label_base_dir: str


@step
def prepare_labels_step(params: DatasetParameters, jsonString: str) -> Output():
    """Convert labelbox json format to VOC format.

    It creates directories at `label_base_dir` that contains VOC format labels.

    Args:
        params (DatasetParameters): parameters for dataset
        jsonString (str): string containing exported labels in json format
    """
    # Convert string from json.dumps to json
    labelbox_export = json.loads(jsonString)

    # Create directories to store converted labels
    create_data_directories(params.label_base_dir)
    logger.info(
        f"Creating directories at {params.label_base_dir} to store labels in VOC format"
    )

    # Parse labels
    annotations = get_annotations(
        params.image_base_dir, params.label_base_dir, labelbox_export
    )
    labels = get_labels(labelbox_export)

    # Save labels to directory in VOC format
    save_annotations_to_xml(params.label_base_dir, annotations)

    save_labels(params.label_base_dir, labels)
