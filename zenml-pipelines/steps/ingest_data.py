"""Collects data from Labelbox using the Labelbox api."""
import os
import json
from PIL import Image
from labelbox import Client
from datetime import datetime

from zenml.steps import Output, step, BaseParameters
from zenml.logger import get_logger

logger = get_logger(__name__)


class IngestDataStepConfig(BaseParameters):
    """Loading Dataset parameters."""

    # Path to image directory containing images
    image_base_dir: str
    # Path to labelbox export json file
    labelbox_export_path: str


@step
def ingest_data(params: IngestDataStepConfig) -> Output():
    """Fetch the data from Labelbox and save it locally.

    Args:
        params (IngestDataStepConfig): parameters for ingesting data
    """
    # Create Labelbox client
    client = Client(api_key=os.environ.get("LABELBOX_API_KEY"))
    # Get project by ID
    project = client.get_project(os.environ.get("LABELBOX_PROJECT_ID"))

    labels = project.label_generator()

    # Get today's date
    end_date = datetime.today().strftime("%Y-%m-%d")
    annotations = project.export_labels(
        download=True, start="1970-01-01", end=end_date
    )

    jsonString = json.dumps(annotations)

    os.makedirs(os.path.dirname(params.image_base_dir), exist_ok=True)

    for label in labels:
        image_name = str(label.data.external_id)
        image_np = label.data.value
        image = Image.fromarray(image_np)
        image.save(f"{params.image_base_dir}{image_name}")

    with open(f"{params.labelbox_export_path}labelbox_export.json", "w") as f:
        f.write(jsonString)
