"""Training step."""
import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

from ..src.train_utils import (
    compute_iou,
    create_confusion_matrix,
    display_and_log_metric,
    test_model,
)

logger = get_logger(__name__)


class EvaluatorParameters(BaseParameters):
    """Evaluate model parameters."""

    # Intersection over Union threshold
    iou_cutoff: float


@step
def evaluate_model(
    params: EvaluatorParameters,
    model: nn.Module,
    test_loader: DataLoader,
    classes: list,
) -> Output():
    """Evaluates the trained model using the test loader dataset.

    Args:
        params (EvaluatorParameters): Parameters for evaluation
        model (nn.Module): trained pytorch object detection model
        test_loader (DataLoader): Test dataloader
        classes (list): Number of unique classes in the dataset
    """
    logger.info(f"Evaluating the trained model.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calculate and return the mAP.
    # Also get the predicted labels and bounding boxes for each image in the test dataset
    test_results, targets, preds = test_model(test_loader, model, device)
    display_and_log_metric(test_results, is_val=False)

    fig = create_confusion_matrix(
        preds=preds,
        targets=targets,
        iou_cutoff=params.iou_cutoff,
        classes=classes,
    )
    mlflow.log_figure(fig, "test_confusion_matrix.png")
