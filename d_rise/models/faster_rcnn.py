from incx.dependencies.d_rise.vision_explanation_methods.explanations import (
    common as od_common,
)
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from incx.models.base_model import BaseModel


class FasterRcnn(BaseModel):

    def __init__(self):
        # Load the pre-trained FasterRCNN model
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(self.device)
        self._model = model

        self._number_of_classes = 95

    def predict(self, x: torch.Tensor):
        
        x = [img.to(self.device) for img in x]

        with torch.no_grad():
            raw_detections = self._model(x)

        def apply_nms(orig_prediction: dict, iou_thresh: float = 0.5):
            keep = torchvision.ops.nms(
                orig_prediction["boxes"], orig_prediction["scores"], iou_thresh
            )

            nms_prediction = orig_prediction
            nms_prediction["boxes"] = nms_prediction["boxes"][keep]
            nms_prediction["scores"] = nms_prediction["scores"][keep]
            nms_prediction["labels"] = nms_prediction["labels"][keep]
            return nms_prediction

        def filter_score(orig_prediction: dict, score_thresh: float = 0.5):
            keep = orig_prediction["scores"] > score_thresh

            filter_prediction = orig_prediction
            filter_prediction["boxes"] = filter_prediction["boxes"][keep]
            filter_prediction["scores"] = filter_prediction["scores"][keep]
            filter_prediction["labels"] = filter_prediction["labels"][keep]
            return filter_prediction

        detections = []
        for raw_detection in raw_detections:
            raw_detection = apply_nms(raw_detection, 0.005)

            # Note that FasterRCNN doesn't return a score for each class, only the predicted class
            # DRISE requires a score for each class. We approximate the score for each class
            # by dividing the (1.0 - class score) evenly among the other classes.

            raw_detection = filter_score(raw_detection, 0.5)
            expanded_class_scores = od_common.expand_class_scores(
                raw_detection["scores"],
                raw_detection["labels"],
                self._number_of_classes,
            )
            detections.append(
                od_common.DetectionRecord(
                    bounding_boxes=raw_detection["boxes"],
                    class_scores=expanded_class_scores,
                    objectness_scores=torch.tensor(
                        [1.0] * raw_detection["boxes"].shape[0]
                    ),
                )
            )

        return detections
