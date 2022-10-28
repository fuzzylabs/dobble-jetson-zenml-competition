"""Training and Evaluation Loops."""
import math

import torch
from rich.pretty import pprint
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from zenml.logger import get_logger

logger = get_logger(__name__)


def train_one_epoch(
    model_name,
    loader: DataLoader,
    net: nn.Module,
    optimizer,
    device,
    print_freq: int,
    epoch: int,
):
    """Train one epoch.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
        ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    FasterRCNN Models: The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    SSDLite Models: The model returns a Dict[Tensor] during training, containing the classification and regression losses.

    Args:
        model_name (str) : Name of the model to be used for training
        loader (DataLoader) : PyTorch Dataloader
        net (nn.Module) : PyTorch Model
        optimizer () : PyTorch optimizer
        device : torch.device indicating whether device is cpu or gpu
        print_freq (int) : Frequency of printing loss statistics
        epochs (int) : Current epoch

    Returns:
        float : Total epoch average loss
    """
    net.train(True)
    num = 0
    total_loss = 0.0
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        # move input and labels to device
        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

        # forward pass
        optimizer.zero_grad()
        loss_dict = net(images, targets)
        # check losses
        losses = sum(loss for loss in loss_dict.values())
        if "fasterrcnn" in model_name:
            regression_loss = loss_dict["loss_box_reg"]
            classification_loss = loss_dict["loss_classifier"]
        elif "ssdlite320" in model_name:
            regression_loss = loss_dict["bbox_regression"]
            classification_loss = loss_dict["classification"]
        loss_value = losses.item()
        num += 1
        # return if loss is nan
        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training")
            logger.info(loss_dict)
            break
        # keep track of losses
        total_loss += loss_value
        running_loss += loss_value
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        # Print loss statistics
        if i and i % print_freq == 0:
            avg_loss = running_loss / print_freq
            avg_reg_loss = running_regression_loss / print_freq
            avg_clf_loss = running_classification_loss / print_freq
            logger.info(
                f"Epoch: {epoch}, Step: {i}/{len(loader)}, "
                + f"Avg Loss: {avg_loss:.4f}, "
                + f"Avg Regression Loss {avg_reg_loss:.4f}, "
                + f"Avg Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

        losses.backward()
        optimizer.step()
    return total_loss / num


def test_one_epoch(loader: DataLoader, net: nn.Module, device):
    """Test one epoch. It prints mAP for dataloader.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
        ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    Args:
        loader (DataLoader): PyTorch dataloder
        net (nn.Module) : PyTorch Model
        device : torch.device indicating whether device is cpu or gpu

    """
    net.eval()
    metric = MeanAveragePrecision()
    for _, data in enumerate(loader):
        # move input and labels to device
        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

        # Print performance statistics
        with torch.no_grad():
            # perform inference
            preds = net(images)
            # get mAP statistics
            metric.update(preds, targets)
            pprint(metric.compute())
