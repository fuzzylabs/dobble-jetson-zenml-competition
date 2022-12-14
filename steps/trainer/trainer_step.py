"""Training step."""
import os

import mlflow
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

from steps.src.train_utils import (
    create_dir,
    display_and_log_metric,
    get_model,
    save_best_model,
    test_one_epoch,
    train_one_epoch,
)

logger = get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class TrainerParameters(BaseParameters):
    """Trainer parameters."""

    # save models folder
    models_folder: str
    # select model from ['fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'ssdlite320_mobilenet_v3_large']
    net: str
    # if True pretrained backbone + pretrained  detection else pretrained backbone only
    use_pretrained: bool
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
    # score threshold to filter bounding boxes
    score_threshold: float
    # save a grid of 3 images (image, ground_truth, predictions) bounding boxes and labels
    save_prediction: bool
    # directory to save images
    prediction_folder: str


def log_params_mlflow(params: TrainerParameters):
    """Log trainer parameters to mlflow.

    Args:
        params (TrainerParameters): Paramters for trainer
    """
    # Log device
    mlflow.log_param("device", device)
    # Log net to use
    mlflow.log_param("Net", params.net)
    # Log model learning rate
    mlflow.log_param("Learning rate", params.lr)
    # Log momentun
    mlflow.log_param("Momentum", params.momentum)
    # Log weight decay
    mlflow.log_param("Weight decay", params.weight_decay)
    # Log t_max value for Cosine Annealing Scheduler
    mlflow.log_param("T-max", params.t_max)
    # Log number of epochs
    mlflow.log_param("Epochs", params.epochs)
    # Log core threshold to filter bounding boxes
    mlflow.log_param("Score threshold", params.score_threshold)


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
    # Check if models folder exists
    create_dir(params.models_folder)
    num_classes = len(classes)
    logger.info(f"Using {params.net} model for training")
    model = get_model(params, num_classes)

    log_params_mlflow(params)

    # Specity the optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay,
    )
    logger.info(f"Using SGD optimizer with learning rate {params.lr}.")

    # Set learning rate policy
    last_epoch = -1
    logger.info("Using CosineAnnealingLR scheduler.")
    scheduler = CosineAnnealingLR(optimizer, params.t_max, last_epoch=last_epoch)  # fmt: skip

    # Train for the desired number of epochs
    logger.info(f"Start training from epoch {last_epoch + 1}.")

    best_map, best_weights, best_epoch = float("-inf"), None, -1
    for epoch in range(last_epoch + 1, params.epochs):
        logger.info("------------------ Training Epoch {} ------------------".format(epoch))  # fmt: skip
        # Training loop
        train_loss = train_one_epoch(
            model_name=params.net,
            loader=train_loader,
            net=model,
            optimizer=optimizer,
            device=device,
            print_freq=params.print_freq,
            epoch=epoch,
        )
        scheduler.step()

        # Evaluate the current model using the validation dataset and return results.
        if params.save_prediction:
            pred_folder = os.path.join(params.prediction_folder, str(epoch))
        # Validation loop
        metric_dict = test_one_epoch(
            loader=val_loader,
            net=model,
            device=device,
            pred_folder=pred_folder,
            score_threshold=params.score_threshold,
            classes=classes,
            save_predictions=params.save_prediction,
        )
        # show metrics output as table
        display_and_log_metric(metric_dict, is_val=True, epoch=epoch)
        curr_epoch_map = metric_dict["map"]
        # save model only if mAP metric has improved
        if curr_epoch_map > best_map:
            best_map, best_epoch, best_weights = save_best_model(
                params=params,
                model=model,
                train_loss=train_loss,
                epoch_map=curr_epoch_map,
                best_map=best_map,
                epoch=epoch,
            )
        logger.info("Finished Training Epoch")
    logger.info(
        f"Loading the best weights from epoch {best_epoch} with map {best_map}"
    )
    # Reinitialize model with best weights
    model.load_state_dict(best_weights)

    # Log Pytorch model to mlflow as artifact
    mlflow.pytorch.log_model(model, "pytorch_model")

    return model
