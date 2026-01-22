# Real-Time License Plate Recognition (ALPR)

This project implements a **CPU-only, real-time license plate recognition (ALPR) system** using a live camera feed. It combines an **ensemble of Faster R-CNN models** for plate detection with **OCR-based plate reading and logging**.

The system is designed to run on a standard PC environment using Python and a virtual environment.

---

## 1. Features

- Live camera capture (webcam / USB camera)
- License plate detection using **5 Faster R-CNN models (ensemble)**
- Confidence filtering and Non-Maximum Suppression (NMS)
- License plate text recognition (OCR)
- Debounced logging of detected plates to file
- CPU-only execution (no GPU / CUDA required)

---

## 2. Project Structure

```
cam_app/
│
├── models/                 # Trained Faster R-CNN model weights (.pth)
│   ├── model_fold_1.pth
│   ├── model_fold_2.pth
│   ├── model_fold_3.pth
│   ├── model_fold_4.pth
│   └── model_fold_5.pth
│
├── src/
│   ├── camera.py           # Camera capture abstraction
│   ├── model_manager.py    # Model loading and management
│   ├── inference.py        # Inference + ensemble logic
│   ├── plate_utils.py      # Plate cropping utilities
│   ├── ocr.py              # OCR logic
│   ├── plate_tracker.py    # Plate debouncing / stabilization
│   └── logger.py           # Plate logging
│
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 3. Requirements

### System Requirements

- Windows (tested)
- Python **3.10 or 3.11** (recommended)
- Webcam or USB camera

### External Dependencies

- **Microsoft Visual C++ Redistributable (2015–2022)**
- **Tesseract OCR** (system installation)

---

## 4. Python Environment Setup

### 4.1 Create Virtual Environment

From the project root directory:

```bat
python -m venv cam_app
cam_app\Scripts\activate
```

Verify:
```bat
python --version
```

---

### 4.2 Install Python Dependencies

Upgrade build tools:

```bat
python -m pip install --upgrade pip setuptools wheel
```

Install requirements:

```bat
pip install -r requirements.txt
```

---

## 5. Install Tesseract OCR

1. Download Tesseract from:
   https://github.com/UB-Mannheim/tesseract/wiki

2. Install and **ensure `tesseract.exe` is added to PATH**

3. Verify installation:

```bat
tesseract --version
```

---

## 6. Model Requirements

- All `.pth` files must be placed in the `./models` directory
- Models must be trained using:

```python
torchvision.models.detection.fasterrcnn_resnet50_fpn
```

- Models must use:

```
num_classes = 2  # background + license plate
```

This value must match the value in `main.py`.

---

## 7. Running the Application

From the project root (with venv activated):

```bat
python main.py
```

Controls:
- Press **`q`** to exit

---

## 8. Output

### Live View

- Green bounding boxes around detected plates
- Plate text rendered above the box

### Logging

Detected plates are logged to:

```
plates.log
```

Format:
```
ISO_TIMESTAMP, PLATE_TEXT
```

Example:
```
2026-01-21T14:32:08.412312, ABC1234
```

Plates are logged only once per cooldown period to avoid duplicates.

---

## 9. Configuration Parameters

Located in `main.py`:

```python
NUM_CLASSES = 2        # Must match training
FRAME_SKIP = 2         # Process every Nth frame
CONF_THRESH = 0.7      # Detection confidence threshold
IOU_THRESH = 0.5       # NMS IoU threshold
```

OCR cooldown can be adjusted in:

```python
PlateTracker(cooldown=3.0)
```

---

## 10. Performance Notes

- Designed for CPU-only execution
- Expected performance: **8–15 FPS** depending on hardware
- OCR is the most expensive step

For better performance:
- Increase `FRAME_SKIP`
- Increase `CONF_THRESH`
- Limit OCR to high-confidence detections

---

## 11. Troubleshooting

### PyTorch DLL Error
- Install Microsoft Visual C++ Redistributable
- Reboot system

### NumPy Build Errors
- Ensure Python 3.10 or 3.11
- Ensure pip is up to date

### No Detections
- Verify `NUM_CLASSES`
- Verify model compatibility
- Check camera index

---

## 12. License

This project is intended for academic and research use.

---

## 13. Acknowledgements

- PyTorch & TorchVision
- OpenCV
- Tesseract OCR

