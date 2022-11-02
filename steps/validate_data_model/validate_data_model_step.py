"""Validate data and model step."""
import mlflow
import torch
from deepchecks.vision.suites import full_suite
from torch import nn
from torch.utils.data import DataLoader
from zenml.steps import Output, step

from steps.src.validate_utils import DobbleData

device = "cuda" if torch.cuda.is_available() else "cpu"


@step
def validate_data_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: nn.Module,
    classes: list,
) -> Output():
    """Perform data and model checks using `full_suite` check function in deepchecks.

    Args:
        train_loader (DataLoader): Pytorch dataloader for the training dataset
        test_loader (DataLoader): Pytorch dataloader for the testing dataset
        model (nn.Module) : Trained Pytorch model
        classes (list) : A list containing unique classes in the dataset

    """
    # LABEL_MAP is a dictionary that maps the class id to the class name
    LABEL_MAP = {i: c for i, c in enumerate(classes)}

    # Create dataset in deepchecks format
    training_data = DobbleData(data_loader=train_loader, label_map=LABEL_MAP)
    test_data = DobbleData(data_loader=test_loader, label_map=LABEL_MAP)
    # Run full_suite checks
    suite = full_suite()
    result = suite.run(training_data, test_data, model, device)
    result.save_as_html("full_suite.html")
    # Log results to Mlflow
    mlflow.log_artifact("full_suite.html", artifact_path="DeepChecksResult")
