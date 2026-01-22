import cv2
import numpy as np

def crop_plate(frame, box, margin=5):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box.tolist())

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return frame[y1:y2, x1:x2]
