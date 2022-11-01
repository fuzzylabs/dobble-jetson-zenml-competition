"""Training and Evaluation Loops."""
import copy
import math
import os
from typing import List

import mlflow
import numpy as np
import torch
from PIL import Image
from rich import print
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes, make_grid
from zenml.logger import get_logger

logger = get_logger(__name__)


def create_dir(folder: str):
    """Create a `folder` directory if it does not exists.

    Args:
        folder (str): Path of the directory to create.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Creating {folder} directory")


def get_image(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Convert normalized tensor to denormalized tensor representing the original image.

    Args:
        x (torch.Tensor): Tensor in range [0-1] used for training
        mean (float): mean used for normalizing input in range [0-1]
        std (float): std used for normalizing input in range [0-1]

    Returns:
        torch.Tensor: tensor in range [0-255] of dtype torch.uint8 representing the image.
    """
    ten = x.clone()
    ten.mul_(std).add_(mean)
    ten = torch.clamp(ten, 0, 1)
    np_arr = (ten * 255.0).numpy().astype(np.uint8)
    return torch.from_numpy(np_arr)


def display_and_log_metric(metrics: dict, epoch: int):
    """Display rich Table output by converting dict to table. It also logs metrics to mlflow.

    Args:
        metrics (dict): Dict containing metrics score for mAP and mAR.
        epoch (int) : Current epoch
    """
    table = Table()
    row = []
    for k, _ in metrics.items():
        table.add_column(k)
    row += [str(np.round(v.item(), 4)) for v in list(metrics.values())]
    table.add_row(*row)
    print(table)
    # log metrics to mlflow
    for k, v in metrics.items():
        mlflow.log_metric(f"val_{k}", v.item(), epoch)


def plot_bounding_box(
    image: torch.Tensor, target: dict, fill: bool, width: int, classes: list
) -> torch.Tensor:
    """Return an tensor (C, H, W) produced by drawing bounding boxes and labels on input image.

    Args:
        image (torch.Tensor): PyTorch Tensor containing image (C, H, W) format (type: torch.uint8)
        target (dict): Dict containing boxes and labels as keys.
                        boxes contains list of tensors (N, 4) and labels contains (N,) indcies corresponding to labels
                        where N is the number of detections.
        fill (bool): whether to fill bounding box with color.
        width (int): width of bounding box to be draw.
        classes (list): list containing unique classes.

    Returns:
        torch.Tensor: tensor in format (C, H, W) containing bounding boxes and labels drawn on the input
    """
    lbl_str = [classes[i] for i in target["labels"]]
    res = draw_bounding_boxes(
        image=image,
        boxes=target["boxes"],
        labels=lbl_str,
        fill=fill,
        width=width,
    )
    return res


def save_grid_predictions(
    image: torch.Tensor,
    gt_im: torch.Tensor,
    pred_im: torch.Tensor,
    filename: str,
):
    """Save 3 images in a grid (image, ground_truth, predictions) to `filename`.

    Args:
        image (torch.Tensor): PyTorch Tensor containing image (C, H, W) format (type: torch.uint8)
        gt_im (torch.Tensor): PyTorch Tensor containing ground truth bounding box (C, H, W) format
        pred_im (torch.Tensor): PyTorch Tensor containing predicted bounding box (C, H, W) format
        filename (str): Path and filename to save grid images
    """
    grid = make_grid([image, gt_im, pred_im])
    grid_image = Image.fromarray(grid.permute(1, 2, 0).numpy().astype(np.uint8))
    grid_image.save(filename, format="PNG")


def plot_and_save_predictions(
    pred_folder: str,
    score_threshold: float,
    images: List[torch.Tensor],
    targets: List[dict],
    preds: List[dict],
    classes: list,
    unique_str: str,
):
    """This function filters the predicted bounding boxes using `score_threshold` and saves a grid of 3 images.

    A grid of (image, ground truth, prediction) is saved at `filename`.

    Args:
        pred_folder (str): Path to predictions folder
        score_threshold (float): score threshold to filter bounding boxes
        images (List[torch.Tensor]): List containing input tensor in range [0-1] format (B, C, h, W)
        targets (List[dict]): A list containing dict of "boxes" and "labels" representing the ground truth
        preds (List[dict]): A list containing dict of "boxes", "scores" and "labels" representing the predictions
        classes (list): list containing unique classes.
        unique_str (str) : a unique string to add to filename to avoid overwriting with same filename
    """
    num = 0
    for img, lbl, pred in zip(images, targets, preds):
        filter_preds = {k: pred[k][pred["scores"] > score_threshold] for k, _ in pred.items()}  # fmt: skip
        if len(filter_preds["boxes"]) > 0:
            # create prediction directory (predictions/epoch) to store grid images
            create_dir(pred_folder)
            # get image from tensor
            image = get_image(img, mean=0, std=1)
            # get bounding box from predictions
            pred_im = plot_bounding_box(
                image=image,
                target=filter_preds,
                fill=True,
                width=4,
                classes=classes,
            )
            # get bounding box from ground truth
            gt_im = plot_bounding_box(
                image=image,
                target=lbl,
                fill=True,
                width=4,
                classes=classes,
            )
            img_path = os.path.join(pred_folder, f"{unique_str}_{num}.png")
            # save 3 images (image, ground_truth, prediction) in a grid
            save_grid_predictions(image, gt_im, pred_im, img_path)
        num += 1


def save_best_model(
    params,
    model: nn.Module,
    train_loss: float,
    epoch_map: float,
    best_map: float,
    epoch: int,
) -> tuple:
    """Save best model only if mAP metric improves on validation dataset.

    Args:
        params (TrainerParameters): Parameters for trainer
        model (nn.Module): Trained pytorch model
        train_loss (float): Current training loss
        epoch_map (float): Current mAP value on validation dataset
        best_map (float): Best mAP value so far on validation dataset
        epoch (int): Current epoch

    Returns:
        tuple: A tuple of best_map, best_epoch and best_weights_
    """
    logger.info(f"mAP metric improved from {best_map} to {epoch_map}")
    model_path = os.path.join(
        params.models_folder,
        f"{params.net}-Epoch-{epoch}-Loss-{train_loss}.pth",
    )
    torch.save(model, model_path)
    logger.info(f"Saved model {model_path}")
    best_map = epoch_map
    best_epoch = epoch
    best_weights = copy.deepcopy(model.state_dict())
    return best_map, best_epoch, best_weights


def train_one_epoch(
    model_name: str,
    loader: DataLoader,
    net: nn.Module,
    optimizer: torch.optim,
    device: str,
    print_freq: int,
    epoch: int,
) -> float:
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
        optimizer (torch.optim) : PyTorch optimizer
        device (str) : string indicating whether device is cpu or gpu
        print_freq (int) : Frequency of printing loss statistics
        epoch (int) : Current epoch

    Returns:
        float : Total epoch average loss
    """
    # set model in training mode
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
        # backward pass
        losses.backward()
        optimizer.step()
        # log training loss to mlflow
        mlflow.log_metric("total_train_loss", total_loss / num, epoch)
    return total_loss / num


def test_one_epoch(
    loader: DataLoader,
    net: nn.Module,
    device: str,
    pred_folder: str,
    score_threshold: float,
    classes: list,
    save_predictions: bool = True,
) -> dict:
    """Test one epoch. It saves the predictions in a grid of (image, ground_truth, prediction) in `predictions/epoch` folder.

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
        device (str) : string indicating whether device is cpu or gpu
        pred_folder (str) :  Path to predictions folder
        score_threshold (float) : Score to filter bounding boxes and labels
        classes (list) : A list containing unique classes
        save_predictions (bool) : save (image, ground_truth and prediction) bounding boxes in a grid

    Returns:
        dict : A dictionary containing Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR) scores.
    """
    # set model in evaluation mode
    net.eval()
    metric = MeanAveragePrecision()
    for i, data in enumerate(loader):
        # move input and labels to device
        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

        # Print performance statistics
        with torch.no_grad():
            # perform inference
            preds = net(images)
            # get mAP statistics
            metric.update(preds, targets)
            # save image, ground_truth and prediction in a grid
            if save_predictions:
                plot_and_save_predictions(
                    pred_folder=pred_folder,
                    score_threshold=score_threshold,
                    images=images,
                    targets=targets,
                    preds=preds,
                    classes=classes,
                    unique_str=str(i),
                )
    return metric.compute()
