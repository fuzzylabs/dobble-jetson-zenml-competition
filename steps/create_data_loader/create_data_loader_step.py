"""Create a data loader."""
import mlflow
from torch.utils.data import DataLoader
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

from steps.src.dobble_dataset import DobbleDataset
from steps.src.transform_utils import get_basic_transform, get_train_transform

logger = get_logger(__name__)


class DataLoaderParameters(BaseParameters):
    """Dataset parameters."""

    # Path to image directory containing images
    dataset_base_dir: str

    # Batch size
    batch_size: int

    # whether to use augmentations
    use_aug: bool

    # image size to use for training
    image_size: int

    # Number of workers for multi-rpocess data loading
    num_workers: int


def log_params_mlflow(
    params: DataLoaderParameters,
    classes: list,
    train_dataset_len: int,
    val_dataset_len: int,
    test_dataset_len: int,
    means: float,
    stds: float,
):
    """Log data loader parameters to mlflow.

    Args:
        params (DataLoaderParameters): Paramters for data loader
        classes (list): List of classes
        train_dataset_len (int): Length of training dataset
        val_dataset_len (int): Length of valdiation dataset
        test_dataset_len (int): Length of testing dataset
        means (float): Means for normalizing inputs in range [0-1]
        stds (float): Stds for normalizing inputs in range [0-1]
    """
    # Log data loader batch size
    mlflow.log_param("Batch size", params.batch_size)
    # Log whether to use augmentations
    mlflow.log_param("Use augmentations", params.use_aug)
    # Log image size to use for training
    mlflow.log_param("Image size", params.image_size)
    # Log number of workers for multi-process data loading
    mlflow.log_param("Num workers", params.num_workers)
    # Log classes
    mlflow.log_param("Classes", classes)
    # Log number of classes
    mlflow.log_param("Num classes", len(classes))
    # Log number of train samples
    mlflow.log_param("Train samples", train_dataset_len)
    # Log number of val samples
    mlflow.log_param("Val samples", val_dataset_len)
    # Log number of test samples
    mlflow.log_param("Test samples", test_dataset_len)
    # Log means
    mlflow.log_param("Mean", means)
    # Log stds
    mlflow.log_param("std", stds)


@step
def create_data_loader(
    params: DataLoaderParameters,
) -> Output(
    train_loader=DataLoader,
    val_loader=DataLoader,
    test_loader=DataLoader,
    classes=list,
):
    """Create PyTorch DataLoader for traiuning, validation and test datasets.

    Args:
        params (DataLoaderParameters): Parameters for DataLoader

    Returns:
        train_loader (DataLoader): Pytorch dataloader for the training dataset
        val_loader (DataLoader): Pytorch dataloader for the validation dataset
        test_loader (DataLoader): Pytorch dataloader for the testing dataset
        classes (list) : A list containing unique classes in the dataset
    """
    # mean and std for normalizing inputs in range [0-1]
    means = 0.0
    stds = 1.0

    # whether to use augmentations
    if params.use_aug:
        logger.info("Using augmentations")
        train_transform = get_train_transform(params.image_size, means, stds)
    else:
        logger.info("Not using augmentations")
        train_transform = get_basic_transform(params.image_size, means, stds)  # fmt: skip
    test_transform = val_transform = get_basic_transform(params.image_size, means, stds)  # fmt: skip
    target_transform = None

    # Prepare train val test datasets
    train_dataset = DobbleDataset(
        root=params.dataset_base_dir,
        transform=train_transform,
        target_transform=target_transform,
        is_test=False,
        is_val=False,
    )
    val_dataset = DobbleDataset(
        root=params.dataset_base_dir,
        transform=val_transform,
        target_transform=target_transform,
        is_test=False,
        is_val=True,
    )
    test_dataset = DobbleDataset(
        root=params.dataset_base_dir,
        transform=test_transform,
        target_transform=target_transform,
        is_test=True,
        is_val=False,
    )

    classes = list(train_dataset.class_names)
    num_classes = len(classes)
    logger.info(f"Number of classes : {num_classes}")
    logger.info(f"Number of samples in train dataset: {len(train_dataset)}")
    logger.info(f"Number of samples in validation dataset: {len(val_dataset)}")
    logger.info(f"Number of samples in test dataset: {len(test_dataset)}")

    # Create training, validation and test dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )

    train_input, train_target = next(iter(train_loader))
    print(f"Input batch shape: {train_input[0].size()}")
    print(f"Bbox batch shape: {train_target[0]['boxes'].size()}")
    print(f"Labels batch shape: {train_target[0]['labels'].size()}")

    log_params_mlflow(
        params,
        classes,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        means,
        stds,
    )

    return train_loader, val_loader, test_loader, classes
