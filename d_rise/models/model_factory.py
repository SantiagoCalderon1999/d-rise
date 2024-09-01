from incx.models.model_enum import ModelEnum
from incx.models.base_model import BaseModel
from incx.models.rt_detr import RtDetr
from incx.models.yolo import Yolo
from incx.models.faster_rcnn import FasterRcnn


class ModelFactory:
    def get_model(self, model: ModelEnum) -> BaseModel:
        if model == ModelEnum.RT_DETR:
            return RtDetr()
        elif model == ModelEnum.YOLO:
            return Yolo()
        elif model == ModelEnum.FASTER_RCNN:
            return FasterRcnn()
        else:
            raise ValueError(f"Unsupported model identifier: {model}")
