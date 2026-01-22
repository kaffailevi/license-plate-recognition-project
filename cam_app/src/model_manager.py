import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class ModelManager:
    def __init__(self, model_dir: str, num_classes: int):
        self.device = "cpu"
        self.models = []
        self.num_classes = num_classes
        self._load_models(model_dir)

    def _load_models(self, model_dir):
        model_paths = sorted(Path(model_dir).glob("*.pth"))

        if not model_paths:
            raise RuntimeError("No .pth models found in ./models")

        for path in model_paths:
            model = get_model(self.num_classes)
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append((path.name, model))

        print(f"Loaded {len(self.models)} Faster R-CNN models")

    def get_models(self):
        return self.models

