from deepchecks.vision.suites import full_suite

import mlflow

import torch
from torch import nn
from torch.utils.data import DataLoader

from deepchecks.core.suite import SuiteResult

from zenml.steps import Output, step

from steps.src.validate_utils import DobbleData

device = "cuda" if torch.cuda.is_available() else "cpu"

@step
def validate_data_model(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, classes: list) -> Output(result=SuiteResult):
    LABEL_MAP = {
        i: c for i, c in enumerate(classes)
    }

    training_data = DobbleData(data_loader=train_loader, label_map=LABEL_MAP)
    test_data = DobbleData(data_loader=test_loader, label_map=LABEL_MAP)

    suite = full_suite()
    result = suite.run(training_data, test_data, model, device=device)
    result.save_as_html("full_suite.html")
    mlflow.log_artifact("full_suite.html", artifact_path="DeepChecksResult")

    return result