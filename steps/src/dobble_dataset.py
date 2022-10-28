"""Dobble Dataset."""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pascal import PascalVOC
from torch.utils.data import Dataset
from zenml.logger import get_logger

logger = get_logger(__name__)


class DobbleDataset(Dataset):
    """Dobble PyTorch Dataset."""

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        is_test: bool = False,
        is_val: bool = False,
        keep_difficult: bool = False,
    ):
        """Class for parsing dataset from VOC format to PyTorch Dataset format.

        Args:
            root (str): the root of the dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages.
            transform: Albumentations Compose function containing train dataset transformations
            target_transform: Albumentations Compose function containing test dataset transformations
            is_test (bool) : parse test dataset
            is_val (bool) : parse validation dataset
            keep_difficult (bool) : whether to use difficult annotations for labels as well
        """
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        # determine the image set file to use
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
            logger.info("Creating Test Dataset")
        elif is_val:
            image_sets_file = self.root / "ImageSets/Main/val.txt"
            logger.info("Creating Validaation Dataset")
        else:
            image_sets_file = self.root / "ImageSets/Main/train.txt"
            logger.info("Creating Training Dataset")

        # get valid image set ID's that contains image and labels
        self.ids = self._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"
        self.class_names = self.get_classes(label_file_name)
        self.class_dict = {
            class_name: i for i, class_name in enumerate(self.class_names)
        }

    def __getitem__(self, index):
        """Returns a sample from the dataset at the given index `idx`.

        Args:
            index (int) : Index int ranging from 0 to len(self.ids)

        Returns:
            tuple: A tuple containing image and target dict containing bounding boxes and labels

        Raises:
            IOError: If unable to read image using `cv2.imread`
        """
        image_id = self.ids[index]
        # get labels, bounding and difficult for particular image_id
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        # get image file corresponding to valid image_id
        image_file = os.path.join(self.root, f"JPEGImages/{image_id}.jpg")
        image = cv2.imread(str(image_file))
        if image is None or image.size == 0:
            raise IOError("failed to load " + str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply transformations
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)  # fmt: skip
            image = transformed["image"]
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)  # fmt: skip
            boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype=torch.int64)  # fmt: skip
            for b, box in enumerate(transformed["bboxes"]):
                boxes_torch[b, :] = torch.tensor(np.round(box))
            boxes = boxes_torch
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        target = {"labels": labels, "boxes": boxes}
        return image, target

    def __len__(self):
        """Number of samples in the dataset.

        Returns:
            int : Total number of samples in dataset
        """
        return len(self.ids)

    def get_classes(self, label_file_name: str):
        """Return a tuple of all classes found in `label.txt` with background class added at 0th index.

        Args:
            label_file_name (str): Path to file named `label.txt`

        Raises:
            FileNotFoundError: If `label.txt` is not found, this function raises an error

        Returns:
            tuple: A tuple of all classes with background class added at 0th index
        """
        if os.path.isfile(label_file_name):
            classes = []

            # classes should be a line-separated list
            with open(label_file_name, "r") as infile:
                for line in infile:
                    classes.append(line.rstrip())

            # prepend BACKGROUND as first class
            classes.insert(0, "BACKGROUND")
            classes = tuple(classes)
            print(f"VOC Labels read from file: {classes}")
            return classes
        else:
            raise FileNotFoundError(f"{label_file_name} does not exist")

    def _read_image_ids(self, image_sets_file: str) -> list:
        """Filter valid image_id by check if following criteria are met by particular image_id.

         1. the number of annotation for particular image id is greater than 0.
         2. if image exists with that particular image_id.

        Args:
            image_sets_file (str): Path to file containing image ids

        Returns:
            list: A list of valid ids
        """
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                image_id = line.rstrip()
                # check if valid image_id
                if len(image_id) <= 0:
                    logger.info("warning - found empty line in {:s}, skipping line".format(image_sets_file))  # fmt: skip
                    continue
                # check if image exists for image_id
                if self._get_num_annotations(image_id) > 0:
                    if self._find_image(image_id):
                        ids.append(line.rstrip())
                    else:
                        logger.info("warning - could not find image {:s} - ignoring from dataset".format(image_id))  # fmt: skip
                else:
                    logger.info("warning - image {:s} has no box/labels annotations, ignoring from dataset".format(image_id))  # fmt: skip
        return ids

    def _get_num_annotations(self, image_id: str) -> int:
        """Total number of annotations for particular image with id : image_id.

        Args:
            image_id (str): image_id for particular image

        Returns:
            int: Number of annotations for particular image
        """
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        ann = PascalVOC.from_xml(annotation_file)
        return len(ann.objects)

    def _get_annotation(self, image_id: str) -> tuple:
        """Get parsed bounding boxes, labels and is_difficult for particular image_id.

        Args:
            image_id (str): image_id for particular image

        Returns:
            tuple: A tuple of 3 numpy array containing parsed bounding boxes, labels and is_difficult
        """
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        ann = PascalVOC.from_xml(annotation_file)
        boxes = []
        labels = []
        is_difficult = []
        for obj in ann.objects:
            class_name = obj.name
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                # VOC dataset format follows Matlab in which indexes start from 1
                x1 = float(obj.bndbox.xmin) - 1
                y1 = float(obj.bndbox.ymin) - 1
                x2 = float(obj.bndbox.xmax) - 1
                y2 = float(obj.bndbox.ymax) - 1
                # append bounding boxes
                boxes.append([x1, y1, x2, y2])
                # append labels
                labels.append(self.class_dict[class_name])
                # parse <difficult> element
                is_difficult_obj = obj.difficult
                is_difficult_str = "0"
                if is_difficult_obj is not None:
                    is_difficult_str = obj.difficult
                is_difficult.append(
                    int(is_difficult_str) if is_difficult_str else 0
                )
            else:
                logger.info("warning - image {:s} has object with unknown class '{:s}'".format(image_id, class_name))  # fmt: skip
        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8),
        )

    def _find_image(self, image_id: str) -> bool:
        """Check if image exists with given image_id.

        Args:
            image_id (str): image_id for particular image

        Returns:
            bool: Returh True if image exists with given image_id else False
        """
        image_file = os.path.join(self.root, f"JPEGImages/{image_id}.jpg")
        if os.path.exists(image_file):
            return True
        return False

    def collate_fn(self, batch):
        """Collation function to convert array and dict returned by dataset to tuple.

        Args:
            batch (_type_): Pytorch dataloader batch

        Returns:
            tuple: A tuple of image and targets (containing dict of bounding boxes and labels)
        """
        return tuple(zip(*batch))
