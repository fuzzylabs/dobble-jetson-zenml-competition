"""Collects data from Labelbox using the Labelbox api."""
import json
import os
from datetime import datetime

from labelbox import Client
from PIL import Image
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

logger = get_logger(__name__)


class IngestDataStepParams(BaseParameters):
    """Loading Dataset parameters."""

    # Path to image directory containing images
    image_base_dir: str


@step
def ingest_data(params: IngestDataStepParams) -> Output(jsonString=str):
    """Fetch the data from Labelbox and save it locally.

    Args:
        params (IngestDataStepParams): parameters for downloading data

    Returns:
        jsonString (str): string containing exported labels in json format
    """
    # check if image folder exists
    if not os.path.exists(params.image_base_dir):
        os.makedirs(params.image_base_dir, exist_ok=True)
        logger.info(
            f"Creating {params.image_base_dir} directory to store images"
        )
    else:
        logger.info(
            f"Reusing existing {params.image_base_dir} directory to store images"
        )

    # Create Labelbox client
    if os.environ.get("LABELBOX_API_KEY") is None:
        logger.error(
            "Labelbox API key not found in environment variables. Please export it using command export LABELBOX_API_KEY='<api-key-here>'"
        )
    else:
        client = Client(api_key=os.environ.get("LABELBOX_API_KEY"))

    # Get project by ID
    if os.environ.get("LABELBOX_PROJECT_ID") is None:
        logger.error(
            "LABELBOX_PROJECT_ID API key not found in environment variables. Please export it using command export LABELBOX_PROJECT_ID='<project-id-here>'"
        )
    else:
        project = client.get_project(os.environ.get("LABELBOX_PROJECT_ID"))
    labels = project.label_generator()

    # Get today's date
    end_date = datetime.today().strftime("%Y-%m-%d")
    # export labels
    annotations = project.export_labels(
        download=True, start="1970-01-01", end=end_date
    )

    # store labels in json format
    root_dir = os.path.dirname(params.image_base_dir)
    labelbox_export_path = os.path.join(root_dir, "labelbox_export.json")
    jsonString = json.dumps(annotations)
    with open(labelbox_export_path, "w") as f:
        f.write(jsonString)
    logger.info(f"Exported labels stored at path: {labelbox_export_path}")

    # save images
    for label in labels:
        image_name = str(label.data.external_id)
        image_np = label.data.value
        image = Image.fromarray(image_np)
        image.save(f"{params.image_base_dir}{image_name}")

    return jsonString
