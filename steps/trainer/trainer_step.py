"""Training step."""
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

from steps.src.train_utils import test_one_epoch, train_one_epoch

logger = get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class TrainerParameters(BaseParameters):
    """Trainer parameters."""

    # save models folder
    models_folder: str
    # select model from ['fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'ssdlite320_mobilenet_v3_large']
    net: str
    # learning rate
    lr: float
    # lr momentum
    momentum: float
    # lr weight decay
    weight_decay: float
    # T_max value for Cosine Annealing Scheduler
    t_max: int
    # number of epochs
    epochs: int
    # print frequency
    print_freq: int


def get_model(params: TrainerParameters, num_classes: int) -> nn.Module:
    """Return a pytorch object detection model.

    Args:
        params (TrainerParameters): Parameters for training
        num_classes (int): Number of classes in the dataset

    Returns:
        nn.Module: pytorch object detection model
    """
    if params.net == "fasterrcnn_mobilenet_v3_large_fpn":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            num_classes=num_classes,
            weights_backbones=MobileNet_V3_Large_Weights.DEFAULT,  # pretrained backbone
            image_mean=[0.485, 0.456, 0.406],  # same mean as backbone
            image_std=[0.229, 0.224, 0.225],  # same std as required by backbone
        )
    if params.net == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes,
            weights_backbones=ResNet50_Weights.DEFAULT,  # pretrained backbone
            image_mean=[0.485, 0.456, 0.406],  # same mean as backbone
            image_std=[0.229, 0.224, 0.225],  # same std as required by backbone
        )
    if params.net == "ssdlite320_mobilenet_v3_large":
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            num_classes=num_classes,
            weights_backbones=MobileNet_V3_Large_Weights.DEFAULT,  # pretrained backbone
        )
    return model


@step
def trainer(
    params: TrainerParameters,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: list,
) -> Output(model=nn.Module):
    """Trains on the train dataloader.

    Args:
        params (TrainerParameters): Parameters for training
        train_loader (DataLoader): Train dataloader
        val_loader (DataLoader): Validation dataloader
        classes (list): Number of unique classes in the dataset

    Returns:
        nn.Module: Trained pytorch  model
    """
    # check if models folder exists
    if not os.path.exists(params.models_folder):
        os.makedirs(params.models_folder, exist_ok=True)
        logger.info(
            f"Creating {params.models_folder} directory to store models"
        )
    else:
        logger.info(
            f"Reusing existing {params.models_folder} directory to store models"
        )
    num_classes = len(classes)
    logger.info(f"Using {params.net} model for training")
    model = get_model(params, num_classes)

    # Specity the optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay,
    )
    logger.info(f"Using SGD optimizer with learning rate {params.lr}.")

    # set learning rate policy
    last_epoch = -1
    logger.info("Using CosineAnnealingLR scheduler.")
    scheduler = CosineAnnealingLR(optimizer, params.t_max, last_epoch=last_epoch)  # fmt: skip

    # train for the desired number of epochs
    logger.info(f"Start training from epoch {last_epoch + 1}.")

    for epoch in range(last_epoch + 1, params.epochs):
        logger.info("------------------ Training Epoch {} ------------------".format(epoch))  # fmt: skip
        train_loss = train_one_epoch(
            params.net,
            train_loader,
            model,
            optimizer,
            device,
            params.print_freq,
            epoch,
        )
        scheduler.step()
        test_one_epoch(val_loader, model, device)

        model_path = os.path.join(
            params.models_folder,
            f"{params.net}-Epoch-{epoch}-Loss-{train_loss}.pth",
        )
        torch.save(model, model_path)
        logger.info(f"Saved model {model_path}")
        logger.info("Finished Training Epoch")
    return model
