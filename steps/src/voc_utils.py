"""Utils for converting to VOC format."""
import os
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image
from zenml.logger import get_logger

logger = get_logger(__name__)
random.seed(42)


@dataclass
class VOCObject:
    """VOCObject class."""

    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Annotation:
    """Annotation class."""

    folder: str
    filename: str
    path: str
    width: int
    height: int
    objects: Sequence[VOCObject]


def get_labels(labelbox_export: list) -> Sequence[str]:
    """Parse labels from json format.

    Args:
        labelbox_export (list): List of dictionary containing annotated labels

    Returns:
        Sequence[str]: Sequence of all unique labels
    """
    labels = []
    for image_id, card in enumerate(labelbox_export):
        if "objects" in card["Label"]:
            for obj in card["Label"]["objects"]:
                labels += [obj["value"]]
    return list(set(labels))


def get_annotations(
    image_base_dir: str, label_base_dir: str, labelbox_export: list
) -> Sequence[Annotation]:
    """Create VOC and Annotation class objects from json format.

    Args:
        image_base_dir (str): Path to directory containing images
        label_base_dir (str): Path to directory to store labels in VOC format
        labelbox_export (list): List of dictionary containing annotated labels

    Returns:
        Sequence[Annotation]: Sequence of Annotation objects for all labels
    """
    annotations = []
    for image_id, card in enumerate(labelbox_export):
        folder = image_base_dir
        filename = card["External ID"]
        _image = Image.open(f"{image_base_dir}/{filename}")
        width, height = (_image.width, _image.height)
        _image.save(f"{label_base_dir}/JPEGImages/{filename}")
        if "objects" in card["Label"]:
            objects = []
            for obj in card["Label"]["objects"]:
                category_name = obj["value"]
                labelbox_bbox = obj["bbox"]
                xmin = labelbox_bbox["left"]
                ymin = labelbox_bbox["top"]
                xmax = xmin + labelbox_bbox["width"]
                ymax = ymin + labelbox_bbox["height"]

                objects += [
                    VOCObject(
                        name=category_name,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                    )
                ]

            annotations += [
                Annotation(
                    folder=folder,
                    filename=filename,
                    path=f"{image_base_dir}/{filename}",
                    width=width,
                    height=height,
                    objects=objects,
                )
            ]
        else:
            continue

    return annotations


def create_data_directories(
    label_base_dir: str,
):
    """Create directories for storing labels.

    Args:
        label_base_dir (str): Path to store labels in VOC format
    """
    os.makedirs(f"{label_base_dir}/Annotations", exist_ok=True)
    os.makedirs(f"{label_base_dir}/ImageSets/Main", exist_ok=True)
    os.makedirs(f"{label_base_dir}/JPEGImages", exist_ok=True)


def voc_object_to_xml(obj: VOCObject) -> str:
    """Create XML using VOCObject.

    Args:
        obj (VOCObject): VOC class object

    Returns:
        str: XML representation of object
    """
    xml = ""
    xml += "<object>\n"
    xml += f"<name>{obj.name}</name>\n"
    xml += "<pose>Unspecified</pose>\n"
    xml += "<truncated>0</truncated>\n"
    xml += "<difficult>0</difficult>\n"

    xml += "<bndbox>\n"
    xml += f"<xmin>{obj.xmin}</xmin>\n"
    xml += f"<ymin>{obj.ymin}</ymin>\n"
    xml += f"<xmax>{obj.xmax}</xmax>\n"
    xml += f"<ymax>{obj.ymax}</ymax>\n"

    xml += "</bndbox>\n"

    xml += "</object>\n"
    return xml


def save_annotations_to_xml(
    label_base_dir: str, annotations: Sequence[Annotation]
):
    """Save the annotations to xml file.

    Args:
        label_base_dir (str): Path to store labels in VOC format
        annotations (Sequence[Annotation]): Sequence of Annotation objects for all labels
    """
    for i, annotation in enumerate(annotations):
        xml = ""
        xml += "<annotation>\n"
        xml += f"<folder>{annotation.folder}</folder>\n"
        xml += f"<filename>{annotation.filename}</filename>\n"
        xml += f"<path>{annotation.path}</path>\n"
        xml += "<source><database>Dobble</database></source>\n"

        xml += "<size>\n"
        xml += f"<width>{annotation.width}</width>\n"
        xml += f"<height>{annotation.height}</height>\n"
        xml += "<depth>3</depth>\n"
        xml += "</size>\n"

        xml += "<segmented>0</segmented>\n"
        for obj in annotation.objects:
            xml += voc_object_to_xml(obj)

        xml += "</annotation>\n"
        with open(
            f"{label_base_dir}/Annotations/{annotation.filename.split('.')[0]}.xml",
            "w",
        ) as f:
            f.write(xml)


def save_labels(label_base_dir: str, labels: Sequence[str]):
    """Save all unique labels to a text file.

    Args:
        label_base_dir (str): Path to store labels in VOC format
        labels (Sequence[str]): Sequence of all unique labels
    """
    with open(f"{label_base_dir}/labels.txt", "w") as f:
        for label in labels:
            f.write(f"{label}\n")


def create_train_test_split(
    split_ratio: float, label_base_dir: str, annotations: Sequence[Annotation]
):
    """Split the dataset into train-val and test dataset in `split_ratio`.

    Args:
        split_ratio (float): A float in range [0, 1] specifying the split ratio
        label_base_dir (str): Path to store labels for both trainval and test dataset in VOC format
        annotations (Sequence[Annotation]): Annotations in VOC format

    Returns:
        train -- a list of image files in the training dataset.
        test -- a list of image files in the testing dataset.
    """
    logger.info(
        f"Splitting the dataset into train {round(split_ratio*100)} %, val dataset {round((1-split_ratio)*100)/2} % and test dataset {round((1-split_ratio)*100)/2} % ratio"
    )

    # Shuffle annotations
    random.shuffle(annotations)
    logger.info(f"Number of labels in dataset: {len(annotations)}")

    # Split annotations in trainval and test dataset
    train, val, test = np.split(
        annotations, [int(len(annotations) * 0.8), int(len(annotations) * 0.9)]
    )
    logger.info(f"Number of labels in train dataset: {len(train)}")
    logger.info(f"Number of labels in val dataset: {len(val)}")
    logger.info(f"Number of labels in test dataset: {len(test)}")

    # Save train labels
    with open(f"{label_base_dir}/ImageSets/Main/train.txt", "w") as f:
        for a in train:
            f.write(a.filename.split(".")[0] + "\n")

    # Save val labels
    with open(f"{label_base_dir}/ImageSets/Main/val.txt", "w") as f:
        for a in val:
            f.write(a.filename.split(".")[0] + "\n")

    # Save test labels
    with open(f"{label_base_dir}/ImageSets/Main/test.txt", "w") as f:
        for a in test:
            f.write(a.filename.split(".")[0] + "\n")

    return trainval, test
