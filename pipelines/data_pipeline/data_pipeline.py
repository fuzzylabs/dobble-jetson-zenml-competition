"""Data pipeline."""
from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline
def data_pipeline(
    ingest_data,
    prepare_labels,
    split_data,
    # validate_data,
    # create_data_release
):
    """Data pipeline.

    Steps
     1. ingest_data: This step fetches the current Dobble data from Labelbox and saves it locally.
     2. prepare_labels: Step for converting labels from labelbox json to VOC format.
     3. split_data: Splits the data into training-validation and testing datasets.
     4. validate_data: Run deepchecks validations on the data, returns True if tests pass.
     5. create_data_release: Adds new dataset to DVC and tags the version with Git.

    Args:
        ingest_data: This step fetches the current Dobble data from Labelbox and saves it locally.
        prepare_labels:  This step converts labels from labelbox json to VOC format.
        split_data : This step splits dataset into train-val and test in `train_test_split_ratio` ratio.
    """
    # specify execution order for the steps
    split_data.after(prepare_labels)

    # Collect data from Labelbox
    labels_json_string = ingest_data()

    # Convert data to VOC format.
    prepare_labels(labels_json_string)

    # Split the data into train-val and test datasets
    split_data(labels_json_string)

    # Run deepchecks on the datasets
    # checks_passed = validate_data()

    # Create a new data release if the tests pass
    # if checks_passed:
    #     create_data_release()
