from typing import Optional

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes: int = 2, pretrained: bool = True):
    weights = "COCO_V1" if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_trained_model(
    weights_path: str, device: Optional[torch.device] = None, num_classes: int = 2
):
    device = device or torch.device("cpu")
    model = create_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
