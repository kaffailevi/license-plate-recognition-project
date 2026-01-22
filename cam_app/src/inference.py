import cv2
import torch
import numpy as np
import torchvision.ops as ops

def ensemble_detections(results, conf_thresh=0.7, iou_thresh=0.5):
    boxes = []
    scores = []

    for _, output in results:
        keep = output["scores"] >= conf_thresh
        boxes.append(output["boxes"][keep])
        scores.append(output["scores"][keep])

    if not boxes or sum(b.shape[0] for b in boxes) == 0:
        return None, None

    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(scores, dim=0)

    keep_idx = ops.nms(boxes, scores, iou_thresh)
    return boxes[keep_idx], scores[keep_idx]

class InferenceEngine:
    def __init__(self, models):
        self.models = models

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    def run(self, frame):
        input_tensor = self.preprocess(frame)

        results = []
        with torch.no_grad():
            for name, model in self.models:
                output = model([input_tensor])[0]
                results.append((name, output))

        return results

