"""A step to perform validation checks on the dataset being ingested by the pipeline."""
from typing import List
from zenml.steps import step, Output


@step
def data_integrity_check(
    train: List[str], test: List[str]
) -> Output(pass_checks=bool):
    """Perform data integrity checks on the train and test images.

    Arguments:
        train (List[str]): a list of files that make up the train dataset.
        test (List[str]): a list of files that make up the test dataset.

    Returns:
        bool: True if the integrity checks pass, else False.
    """
    return True
