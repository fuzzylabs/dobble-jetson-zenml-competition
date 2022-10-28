"""Upload the data folder generated in the data pipeline to a S3 bucket."""
import os

import boto3
from botocore.exceptions import ClientError
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

logger = get_logger(__name__)


class UploadDataParameters(BaseParameters):
    """Upload Dataset parameters."""

    # Path to the data folder containing all outputs from the data pipeline
    data_base_dir: str
    # Name of the aws service
    service_name: str
    # Name of the S3 bucket
    bucket: str


@step
def upload_data(params: UploadDataParameters) -> Output():
    """Upload the data folder onto a S3 bucket.

    Args:
        params (UploadDataParameters): Parameters for uploading data
    """
    access_key = os.environ.get("ACCESS_KEY_ID")
    secret_key = os.environ.get("SECRET_ACCESS_KEY")
    logger.info(f"Creating a S3 service client.")
    s3_client = boto3.client(
        params.service_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    logger.info(f"Client created")

    try:
        for roots, _, files in os.walk(params.data_base_dir):
            s3_client.put_object(Bucket=params.bucket, Key=f"{roots}/")
            for file in files:
                file_path = f"{roots}/{file}"
                s3_client.upload_file(file_path, params.bucket, file_path)
        logger.info(f"All data uploaded to S3 bucket")
    except ClientError as e:
        logger.error(e)
