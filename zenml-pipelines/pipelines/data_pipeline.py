"""Data pipeline."""
from zenml.pipelines import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def data_pipeline(prepare_labels):
    """Create data pipeline.

    Args:
        prepare_labels: Step for converting labels from labelbox json to VOC format.
    """
    # step to convert labels from labelbox json to VOC format
    prepare_labels()
