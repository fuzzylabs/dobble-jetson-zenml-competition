import numpy as np
import torch
import torchvision
from deepchecks.vision.detection_data import DetectionData


class DobbleData(DetectionData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_to_images(self, batch):
        """
        Convert a batch of data to images in the expected format. The expected format is an iterable of cv2 images,
        where each image is a numpy array of shape (height, width, channels). The numbers in the array should be in the
        range [0, 255] in a uint8 format.
        """
        inp = torch.stack(list(batch[0])).cpu().detach().numpy().transpose((0, 2, 3, 1))
        mean = 0
        std = 1
        # Un-normalize the images
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp * 255

    def batch_to_labels(self, batch):
        """
        Convert a batch of data to labels in the expected format. The expected format is a list of tensors of length N,
        where N is the number of samples. Each tensor element is in a shape of [B, 5], where B is the number of bboxes
        in the image, and each bounding box is in the structure of [class_id, x, y, w, h].
        """
        tensor_annotations = batch[1]
        label = []
        for annotation in tensor_annotations:
            if len(annotation["boxes"]):
                bbox = annotation["boxes"]
                # Convert the Pascal VOC xyxy format to xywh format
                bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
                # The label shape is [class_id, x, y, w, h]
                label.append(
                    torch.concat([torch.reshape(annotation["labels"], (-1, 1)), bbox], dim=1)
                )
            else:
                # If it's an empty image, we need to add an empty label
                label.append(torch.tensor([]))
        return label

    def infer_on_batch(self, batch, model, device):
        """
        Returns the predictions for a batch of data. The expected format is a list of tensors of shape length N, where N
        is the number of samples. Each tensor element is in a shape of [B, 6], where B is the number of bboxes in the
        predictions, and each bounding box is in the structure of [x, y, w, h, score, class_id].
        """
        nm_thrs = 0.2
        score_thrs = 0.7
        imgs = list(img.to(device) for img in batch[0])
        # Getting the predictions of the model on the batch
        with torch.no_grad():
            preds = model(imgs)
        processed_pred = []
        for pred in preds:
            # Performoing non-maximum suppression on the detections
            keep_boxes = torchvision.ops.nms(pred['boxes'], pred['scores'], nm_thrs)
            score_filter = pred['scores'][keep_boxes] > score_thrs

            # get the filtered result
            test_boxes = pred['boxes'][keep_boxes][score_filter].reshape((-1, 4))
            test_boxes[:, 2:] = test_boxes[:, 2:] - test_boxes[:, :2]  # xyxy to xywh
            test_labels = pred['labels'][keep_boxes][score_filter]
            test_scores = pred['scores'][keep_boxes][score_filter]

            processed_pred.append(
                torch.concat([test_boxes, test_scores.reshape((-1, 1)), test_labels.reshape((-1, 1))], dim=1))

        return processed_pred
