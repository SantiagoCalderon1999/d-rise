from d_rise.models.model_enum import ModelEnum
from d_rise.models.base_model import BaseModel
from d_rise.models.rt_detr import RtDetr
from d_rise.models.yolo import Yolo
from d_rise.models.faster_rcnn import FasterRcnn


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
