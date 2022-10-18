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

    # Path to labelbox export json file
    labelbox_export_path: str


@step
def prepare_labels_step(params: DatasetParameters) -> Output():
    """Convert labelbox json format to VOC format.

    Args:
        params (DatasetParameters): parameters for dataset
    """
    with open(params.labelbox_export_path) as f:
        labelbox_export = json.load(f)

    # create directories to store converted labels
    create_data_directories(params.label_base_dir)

    # parse labels
    annotations = get_annotations(
        params.image_base_dir, params.label_base_dir, labelbox_export
    )
    labels = get_labels(labelbox_export)

    # save labels to directory in VOC format
    save_annotations_to_xml(params.label_base_dir, annotations)

    save_labels(params.label_base_dir, labels)
