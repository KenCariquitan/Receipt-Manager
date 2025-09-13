# app/detect.py
from ultralytics import YOLO
import cv2
from pathlib import Path

# Weights location (you copied best.pt here)
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo_receipt.pt"

# Change these names to exactly match yolo_data/data.yaml → names: [...]
CLASSES = ["Date", "Merchant", "Total"]

_model = None

def get_model():
    """Lazy-load YOLO model once."""
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = YOLO(str(MODEL_PATH))
    return _model

def detect_fields(img_bgr, conf: float = 0.20, imgsz: int = 960):
    """
    Run YOLO on a BGR image and return a list of detections:
    [{"name": <class_name>, "box": (x1,y1,x2,y2), "conf": float}, ...]
    Safe against None/empty results.
    """
    m = get_model()
    if m is None:
        # Model weights not found or failed to load
        return []

    # YOLO expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Ultralytics returns a list-like of Results; take first image result
    results = m.predict(source=img_rgb, imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []

    res = results[0]

    # Guard: boxes may be missing or empty
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return []

    try:
        n = len(boxes)  # Boxes supports __len__
    except Exception:
        return []

    if n == 0:
        return []

    H, W = img_bgr.shape[:2]
    out = []
    # Access tensors safely by index; avoid iterating "boxes" directly for type-checkers
    for i in range(n):
        try:
            cls_id_t = boxes.cls[i]
            conf_t = boxes.conf[i]
            xyxy_t = boxes.xyxy[i]

            cls_id = int(cls_id_t.item())
            if not (0 <= cls_id < len(CLASSES)):
                continue

            x1, y1, x2, y2 = map(int, xyxy_t.tolist())
            # clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            out.append({
                "name": CLASSES[cls_id],
                "box": (x1, y1, x2, y2),
                "conf": float(conf_t.item()),
            })
        except Exception:
            # If any single box is malformed, skip it and continue
            continue

    return out