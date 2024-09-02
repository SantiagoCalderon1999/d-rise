# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Implementation of DRISE.

A black box explainability method for object detection.
"""

import base64
import copy
import io
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple

import pandas as pd
import PIL.Image as Image
import torch
import torchvision.transforms as T
import tqdm
import random

from torch import Tensor
from .common import (DetectionRecord, GeneralObjectDetectionModelWrapper,
                     compute_affinity_matrix)


@dataclass
class MaskAffinityRecord:
    """Class for keeping track of masks and associated affinity score.

    :param mask: 3xHxW mask
    :type mask: torch.Tensor
    :param affinity_scores: Scores for each detection in each image
        associated with mask.
    :type affinity_scores: List of Tensors
    """

    def __init__(
            self,
            mask: torch.Tensor,
            affinity_scores: List[torch.Tensor]
    ):
        """Initialize the MaskAffinityRecord."""
        self.mask = mask
        self.affinity_scores = affinity_scores

    def get_weighted_masks(self) -> List[torch.Tensor]:
        """Return the masks weighted by the affinity scores.

        :return: Masks weighted by affinity scores - N tensors of shape
            Dx3xHxW, where N is the number of images in the batch, D, is the
            number of detections in an image (where D changes image to image)
        :rtype: List of Tensors
        """
        weighted_masks = []
        for image_affinity_scores in self.affinity_scores:
            weighted_masks.append(
                image_affinity_scores.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                * self.mask
            )

        return weighted_masks

    def to(self, device: str):
        """Move affinity record to accelerator.

        :param device: Torch string describing device, e.g. 'cpu' or 'cuda:0'
        :type device: String
        """
        self.mask = self.mask.to(device)
        device_affinity_scores = []
        for score in self.affinity_scores:
            # note: this does not update the original list
            score = score.to(device)
            device_affinity_scores.append(score)
        self.affinity_scores = device_affinity_scores


def generate_mask(
        base_size: Tuple[int, int],
        img_size: Tuple[int, int],
        padding: int,
        device: str,
        seed: float
) -> torch.Tensor:
    """Create a random mask for image occlusion.

    :param base_size: Lower resolution mask grid shape
    :type base_size: Tuple (int, int)
    :param img_size: Size of image to be masked (hxw)
    :type img_size: Tuple (int, int)
    :param padding: Amount to offset mask
    :type padding: int
    :param device: Torch string describing device, e.g. 'cpu' or 'cuda:0'
    :type device: String
    :return: Occlusion mask for image, same shape as image
    :rtype: Tensor
    """
    # Needs to be float for resize interpolation
    torch.manual_seed(seed)
    base_mask = 1.0 * torch.randint(0, 2, base_size, device=device)
    mask = base_mask.repeat([3, 1, 1])
    resized_mask = T.Resize(
        (img_size[0] + padding, img_size[1] + padding),
        # Interpolation mode makes a BIG difference for occlusion maps
        interpolation=Image.NEAREST
    )(mask)

    return T.RandomCrop(img_size)(resized_mask)


def fuse_mask(
        img_tensor: torch.Tensor,
        mask: torch.Tensor
) -> torch.Tensor:
    """Mask an image tensor.

    :param img_tensor: Image to be masked
    :type img_tensor: Tensor
    :param mask: Mask for image
    :type mask: Tensor
    :return: Masked image
    :rtype: Tensor
    """
    return img_tensor * mask


def compute_affinity_scores(
    base_detections: DetectionRecord,
    masked_detections: DetectionRecord
) -> torch.Tensor:
    """Compute highest affinity score between two sets of detections.

    :param base_detections: Set of detections to get affinity scores for
    :type base_detections: Detection Record
    :param masked_detections: Set of detections to score against
    :type masked_detections: Detection Record
    :return: Set of affinity scores associated with each detections
    :rtype: Tensor of shape D, where D is number of base detections
    """
    score_matrix = compute_affinity_matrix(base_detections, masked_detections)
    return torch.max(score_matrix, dim=1)[0]


def saliency_fusion(
        affinity_records: List[MaskAffinityRecord],
        device: str,
        normalize: Optional[bool] = True,
        verbose: bool = False
) -> torch.Tensor:
    """Create a fused mask based on the affinity scores of the different masks.

    :param affinity_records: List of affinity records computed for mask
    :type affinity_records: List of affinity records
    :param device: Torch string describing device, e.g. 'cpu' or 'cuda:0'
    :type device: String
    :param normalize: Normalize the image by subtracting off the average
        affinity score (optional), defaults to true
    :type: bool
    :return: List of saliency maps - one list of maps for each image in batch,
        and one map per detection in each image
    :rtype: List of Tensors - one tensor for each image, and each tensor of
        shape Dx3xHxW, where D is the number of detections in that image.
    """
    average_scores_accum = copy.deepcopy(affinity_records[0].affinity_scores)
    weighted_masks_accum = copy.deepcopy(
        affinity_records[0].get_weighted_masks())
    unweighted_mask_accum = copy.deepcopy(affinity_records[0].mask)

    records_iterator = tqdm.tqdm(affinity_records[1:]) if verbose \
        else affinity_records[1:]

    for affinity_record in records_iterator:
        try:
            unweighted_mask_accum += affinity_record.mask
            affinity_scores = affinity_record.affinity_scores
            weighted_masks = affinity_record.get_weighted_masks()

            for weighted_mask_accum, weighted_mask in zip(weighted_masks_accum,
                                                          weighted_masks):
                weighted_mask_accum += weighted_mask

            for average_score_accum, affinity_score in zip(
                    average_scores_accum,
                    affinity_scores):
                average_score_accum += affinity_score
        except RuntimeError:
            continue

    num_affinity_records = len(affinity_records)
    for scores in average_scores_accum:
        scores /= num_affinity_records

    if normalize:
        for average_score, weighted_mask in zip(average_scores_accum,
                                                weighted_masks_accum):
            weighted_mask -= average_score.unsqueeze(1).unsqueeze(1). \
                unsqueeze(1) * unweighted_mask_accum

    normalized_masks = []

    for imgs in weighted_masks_accum:
        normed_masks = []
        for mask in imgs:
            mask = mask - torch.min(mask)
            mask = mask / torch.max(mask)
            normed_masks.append({'detection': mask})
        normalized_masks.append(normed_masks)

    return normalized_masks
import matplotlib.pyplot as plt
import numpy as np

def DRISE_saliency(
        model: GeneralObjectDetectionModelWrapper,
        image_tensor: Tensor,
        target_detections: List[DetectionRecord],
        number_of_masks: int,
        mask_res: Tuple[int, int] = (16, 16),
        mask_padding: Optional[int] = None,
        device: str = "cpu",
        verbose: bool = False,
        seed: int = 0
) -> List[torch.Tensor]:
    """Compute DRISE saliency map.

    :param model: Object detection model wrapped for occlusion
    :type model: OcclusionModelWrapper
    :param target_detections: Baseline detections to get saliency
        maps for
    :type target_detections: List of Detection Records
    :param number_of_masks: Number of masks to use for saliency
    :type number_of_masks: int
    :param mask_res: Resolution of mask before scale up
    :type maks_res: Tuple of ints
    :param mask_padding: How much to pad the mask before cropping
    :type: Optional int
    :device: Device to use to run the function
    :type: str
    :return: A list of tensors, one tensor for each image. Each tensor
        is of shape [D, 3, W, H], and [i ,3 W, H] is the saliency map
        associated with detection i.
    :rtype: List torch.Tensor
    """
    img_size = image_tensor.shape[-2:]
    if mask_padding is None:
        mask_padding = int(max(
            img_size[0] / mask_res[0], img_size[1] / mask_res[1]))

    mask_records = []

    mask_iterator = tqdm.tqdm(range(number_of_masks)) if verbose \
        else range(number_of_masks)
    random.seed(seed)
    for _ in mask_iterator:
        mask_seed = random.randint(1, 1000000)
        mask = generate_mask(mask_res, img_size, mask_padding, device, seed = mask_seed)
        masked_image = fuse_mask(image_tensor, mask)
        with torch.no_grad():
            masked_detections = model.predict(masked_image)
        # print(masked_detections[0].class_scores)
        # max_vals = [max(class_score) for class_score in masked_detections[0].class_scores]
        # plt.title(f'{max_vals}')
        # plt.imshow(tensor_to_numpy_image(masked_image))
        # plt.show()
        affinity_scores = []

        for (target_detection, masked_detection) in zip(target_detections,
                                                        masked_detections):
            affinity_scores.append(
                compute_affinity_scores(target_detection,
                                        masked_detection).detach().to("cpu"))
        mask_records.append(MaskAffinityRecord(
            mask=mask.to("cpu"),
            affinity_scores=affinity_scores)
        )
    return saliency_fusion(mask_records, device, verbose=verbose)

def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array suitable for plt.imshow.

    :param tensor: A tensor of shape (C, H, W) with values in the range [0, 1] or [0, 255]
    :return: A NumPy array of shape (H, W, C) with values in the range [0, 255]
    """
    if len(tensor.shape) == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # Remove the batch dimension if it exists

    if tensor.dtype == torch.float32:
        tensor = tensor * 255  # Convert [0, 1] range to [0, 255]

    tensor = tensor.byte()  # Convert to byte tensor

    if tensor.shape[0] == 1:  # Grayscale image
        tensor = tensor.squeeze(0)
        numpy_image = tensor.cpu().numpy()
    elif tensor.shape[0] == 3:  # RGB image
        tensor = tensor.permute(1, 2, 0)  # Change to (H, W, C)
        numpy_image = tensor.cpu().numpy()
    else:
        raise ValueError("Tensor shape not supported for image conversion")

    return numpy_image

def convert_base64_to_tensor(b64_img: str, device: str) -> Tensor:
    """Convert base64 image to tensor.

    :param b64_img: Base64 encoded image
    :type b64_img: str
    :param device: Torch string describing device, e.g. "cpu" or "cuda:0"
    :type device: str
    :return: Image tensor
    :rtype: Tensor
    """
    base64_decoded = base64.b64decode(b64_img)
    image = Image.open(io.BytesIO(base64_decoded))
    img_tens = T.ToTensor()(image).to(device)
    return img_tens


def convert_tensor_to_base64(img_tens: Tensor) -> Tuple[str, Tuple[int, int]]:
    """Convert image tensor to base64 string.

    :param img_tens: Image tensor
    :type img_tens: Tensor
    :return: Base64 encoded image
    :rtype: str
    """
    img_pil = T.ToPILImage()(img_tens)
    imgio = BytesIO()
    img_pil.save(imgio, format='PNG')
    img_str = base64.b64encode(imgio.getvalue()).decode('utf8')
    return img_str, img_pil.size


def DRISE_saliency_for_mlflow(
        model,
        image_tensor: pd.DataFrame,
        target_detections: List[DetectionRecord],
        number_of_masks: int,
        mask_res: Tuple[int, int] = (16, 16),
        mask_padding: Optional[int] = None,
        device: str = "cpu",
        verbose: bool = False,
) -> List[torch.Tensor]:
    """Compute DRISE saliency map.

    :param model: Object detection model wrapped for occlusion
    :type model: OcclusionModelWrapper
    :param target_detections: Baseline detections to get saliency
        maps for
    :type target_detections: List of Detection Records
    :param number_of_masks: Number of masks to use for saliency
    :type number_of_masks: int
    :param mask_res: Resolution of mask before scale up
    :type maks_res: Tuple of ints
    :param mask_padding: How much to pad the mask before cropping
    :type: Optional int
    :device: Device to use to run the function
    :type: str
    :return: A list of tensors, one tensor for each image. Each tensor
        is of shape [D, 3, W, H], and [i ,3 W, H] is the saliency map
        associated with detection i.
    :rtype: List torch.Tensor
    """
    if not isinstance(image_tensor, pd.DataFrame):
        raise ValueError(
            "TypeError: Image needs to be a torch.Tensor or pd.DataFrame")
    if not image_tensor.shape[0] == 1:
        raise ValueError(
            "Currently only one image supported for AutoML mlflow model")

    img_size = image_tensor.loc[0, 'image_size']

    if mask_padding is None:
        mask_padding = int(max(
            img_size[0] / mask_res[0], img_size[1] / mask_res[1]))

    mask_records = []

    mask_iterator = tqdm.tqdm(range(number_of_masks)) if verbose \
        else range(number_of_masks)

    # Currently only supports single image
    img_tens = convert_base64_to_tensor(
        image_tensor.loc[0, 'image'], device)

    for _ in mask_iterator:
        # Converts image base64 to a tensor
        # Fuses mask tensor with image tensor
        # Converts fused image tensor to base64
        mask = generate_mask(mask_res, img_size, mask_padding, device)

        masked_image = fuse_mask(img_tens, mask)

        masked_image_str, masked_image_size = convert_tensor_to_base64(
            masked_image)

        masked_df = pd.DataFrame(
            data=[[masked_image_str, masked_image_size]],
            columns=['image', "image_size"],
        )

        masked_detections = model.predict(masked_df)

        affinity_scores = [
            compute_affinity_scores(target_detection, masked_detection)
            for (target_detection, masked_detection)
            in zip(target_detections, masked_detections)
        ]

        mask_records.append(MaskAffinityRecord(
            mask=mask.detach().cpu(),
            affinity_scores=[s.detach().cpu() for s in affinity_scores])
        )
    return saliency_fusion(mask_records, device, verbose=verbose)
