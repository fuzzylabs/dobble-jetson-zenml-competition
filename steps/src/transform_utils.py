"""Data Augmentation utils."""
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


def get_basic_transform(image_size: int, means: float, stds: float):
    """A common transformation function containing basic transformation such as resize and normalize.

    Args:
        image_size (int): Size of square image (both height and width are same)
        means (float): A float containing  mean to convert image in range [0-1]
        stds (float): A float contaitning standard deviation to convert image in range [0-1]

    Returns:
        albumentations.core.composition.Compose: Albumentations transformed function that takes input and returns transformed version as output
    """
    transform = A.Compose(
        [
            A.Resize(
                width=image_size,
                height=image_size,
            ),
            A.Normalize(mean=means, std=stds),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    return transform


def get_train_transform(image_size: int, means: float, stds: float):
    """A transformation function used for training containing augmentations such as flips, distort, rotate, etc.

    Args:
        image_size (int): Size of square image (both height and width are same)
        means (float): A float containing  mean to convert image in range [0-1]
        stds (float): A float contaitning standard deviation to convert image in range [0-1]

    Returns:
        albumentations.core.composition.Compose: Albumentations transformed function that takes input and returns transformed version as output
    """
    transform = A.Compose(
        [
            A.OpticalDistortion(),
            A.RandomSizedBBoxSafeCrop(
                height=int(image_size / 3), width=int(image_size / 3)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.SafeRotate([-90, 90], p=1),
            A.Resize(
                width=image_size,
                height=image_size,
            ),
            A.Normalize(mean=means, std=stds),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    return transform
