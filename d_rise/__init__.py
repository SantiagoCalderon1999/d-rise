# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for creating explanations for vision models."""

from .version import name, version
from .DRISE_runner import get_saliency_map
from .models.yolo import YOLO
from .models.rt_detr import RTDETR
from .models.faster_rcnn import FasterRcnn
from .models.model_factory import ModelFactory
from .models.model_enum import ModelEnum

__name__ = name
__version__ = version
