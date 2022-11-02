"""A step to perform validation checks on the dataset being ingested by the pipeline."""
import torch

from torch.utils.data import DataLoader
from zenml.steps import BaseParameters, Output, step
from steps.src.train_utils import get_model

from steps.src.validate_utils import DobbleData

device = "cuda" if torch.cuda.is_available() else "cpu"


class DataValidationParameters(BaseParameters):
    """Data validation parameters."""

    # select model from ['fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'ssdlite320_mobilenet_v3_large']
    net: str
    # if True pretrained backbone + pretrained  detection else pretrained backbone only
    use_pretrained: bool

@step
def data_integrity_check(
    params: DataValidationParameters,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    classes: list,
) -> Output():
    """Perform data integrity checks on the train and test images.

    Arguments:
        train (List[str]): a list of files that make up the train dataset.
        test (List[str]): a list of files that make up the test dataset.

    Returns:
        bool: True if the integrity checks pass, else False.
    """
    # We have a single label here, which is the tomato class
    # The label_map is a dictionary that maps the class id to the class name, for display purposes.
    LABEL_MAP = {
        i: c for i, c in enumerate(classes)
    }

    model = get_model(params, len(classes))
    model.eval()

    training_data = DobbleData(data_loader=train_loader, label_map=LABEL_MAP)
    val_data = DobbleData(data_loader=val_loader, label_map=LABEL_MAP)
    test_data = DobbleData(data_loader=test_loader, label_map=LABEL_MAP)

    training_data.validate_format(model, device=device)
    val_data.validate_format(model, device=device)
    test_data.validate_format(model, device=device)