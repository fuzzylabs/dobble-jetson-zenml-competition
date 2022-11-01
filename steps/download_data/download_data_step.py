"""Download the data from the S3 bucket (generated in the data pipeline)."""
import os
from typing import Tuple

import boto3 as aws_conn
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

logger = get_logger(__name__)


class DownloadDataParameters(BaseParameters):
    """The parameters required for downloading the data from S3."""

    # the path for where the data should be downloaded to
    dataset_base_dir: str

    # the specific AWS service that we're accessing
    service_name: str

    # the name of the bucket where the data is stored
    bucket_name: str


def get_path_and_file(full_file_path: str) -> Tuple[str, str]:
    """A utility function to get the file path and file name from a path string.

    Arguments:
        full_file_path (str): the full path for the file stored remotely.

    Returns:
        Tuple[str, str]: the file path and file name.
    """
    file_path = os.path.dirname(full_file_path)
    file_name = os.path.basename(full_file_path)

    return file_path, file_name


@step
def download_data(params: DownloadDataParameters) -> Output():
    """A step which downloads the data required for the training pipeline from AWS.

    Args:
        params (DownloadDataParameters): the parameters required to download the data from AWS.
    """
    logger.info("Downloading data from S3 bucket...")

    # if the dataset base directory doesn't exist, create it
    if not os.path.isdir(params.dataset_base_dir):
        os.mkdir(params.dataset_base_dir)

    aws_access_key_id = os.environ.get("ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("SECRET_ACCESS_KEY")

    if (aws_access_key_id is None) or (aws_secret_access_key is None):
        logger.error(
            "AWS API keys not found in the environment variables. Please set both the ACCESS_KEY_ID and SECRET_ACCESS_KEY"
        )

    # set up the AWS connection session
    aws_session = aws_conn.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # access the bucket where the data is stored
    s3_resource = aws_session.resource("s3")
    bucket = s3_resource.Bucket(params.bucket_name)

    # get the set of files which have the dataset_base_dir as the prefix
    files_to_download = list(
        bucket.objects.filter(Prefix=params.dataset_base_dir)
    )

    # for each of those files, download them
    for each_file in files_to_download:
        f_path, f_name = get_path_and_file(each_file.key)

        # create the directory if it doesn't already exist
        if not os.path.isdir(f_path):
            os.mkdir(f_path)

        if f_name:
            bucket.download_file(each_file.key, each_file.key)

    logger.info(f"Data downloaded to {params.dataset_base_dir}...")
