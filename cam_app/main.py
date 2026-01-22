import cv2
import torch

from src.camera import Camera
from src.model_manager import ModelManager
from src.inference import InferenceEngine, ensemble_detections
from src.plate_utils import crop_plate
from src.ocr import read_plate
from src.plate_tracker import PlateTracker
from src.logger import log_plate


MODEL_DIR = "./models"
NUM_CLASSES = 2          # background + license plate
FRAME_SKIP = 2
CONF_THRESH = 0.7
IOU_THRESH = 0.5

tracker = PlateTracker(cooldown=4.0)


def main():
    torch.set_num_threads(4)

    camera = Camera()
    model_manager = ModelManager(MODEL_DIR, NUM_CLASSES)
    inference_engine = InferenceEngine(model_manager.get_models())

    frame_id = 0
    results = None  # keep last results for skipped frames

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # ---------- 1. Inference ----------
        if frame_id % FRAME_SKIP == 0:
            results = inference_engine.run(frame)

        # ---------- 2. Ensemble + draw ----------
        if results is not None:
            boxes, scores = ensemble_detections(
                results,
                conf_thresh=CONF_THRESH,
                iou_thresh=IOU_THRESH
            )

            if boxes is not None:
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = map(int, box.tolist())

                    plate_img = crop_plate(frame, box)
                    plate_text = read_plate(plate_img)

                    if plate_text and tracker.should_log(plate_text, score):
                        log_plate(plate_text)
                        print(f"Detected plate: {plate_text}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if plate_text:
                        cv2.putText(
                            frame,
                            plate_text,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

        # ---------- 3. Display ----------
        cv2.imshow("Live License Plate Detection", frame)

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
