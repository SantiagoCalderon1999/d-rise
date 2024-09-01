from incx.dependencies.d_rise.vision_explanation_methods.explanations import (
    common as od_common,
)
import torch
from typing import List
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def predict(self, x: torch.Tensor) -> List[od_common.DetectionRecord]:
        """Take a tensor and return a list of detection records.

        This is the only required method.

        :param x: Tensor of a batch of images. Shape [B, 3, W, H]
        :type x: torch.Tensor
        :return: List of Detections produced by wrapped model
        :rtype: List of DetectionRecords
        """
        raise NotImplementedError
