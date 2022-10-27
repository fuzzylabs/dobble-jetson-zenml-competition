"""Create a data loader."""
import numpy as np
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


@step
def create_data_loader(
    params: DataLoaderParameters,
) -> Output(
    train_loader=DataLoader, val_loader=DataLoader, test_loader=DataLoader
):
    """Create PyTorch DataLoader for traiuning, validation and test datasets.

    Args:
        params (DataLoaderParameters): Parameters for DataLoader

    Returns:
        train_loader (DataLoader): Pytorch dataloader for the training dataset
        val_loader (DataLoader): Pytorch dataloader for the validation dataset
        test_loader (DataLoader): Pytorch dataloader for the testing dataset
    """
    # mean and std for normalizing inputs
    means = np.array([127, 127, 127])  # RGB layout
    stds = 128.0

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

    num_classes = len(train_dataset.class_names)
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
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=False,
    )

    train_input, train_bbox, train_labels = next(iter(train_loader))
    logger.info(f"Input batch shape: {train_input.size()}")
    logger.info(f"Bbox batch shape: {train_bbox.size()}")
    logger.info(f"Labels batch shape: {train_labels.size()}")

    return train_loader, val_loader, test_loader