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

def detect_fields(img_bgr, conf: float = 0.15, imgsz: int = 1280):
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
    raw: list[dict] = []
    for i in range(n):
        try:
            cls_id_t = boxes.cls[i]
            conf_t = boxes.conf[i]
            xyxy_t = boxes.xyxy[i]

            cls_id = int(cls_id_t.item())
            if not (0 <= cls_id < len(CLASSES)):
                continue

            x1, y1, x2, y2 = map(int, xyxy_t.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            raw.append({
                "name": CLASSES[cls_id],
                "box": (x1, y1, x2, y2),
                "conf": float(conf_t.item()),
            })
        except Exception:
            continue

    # Merge overlapping boxes by class, keeping the highest confidence
    merged: list[dict] = []
    for det in raw:
        overlaps = [
            idx for idx, existing in enumerate(merged)
            if existing["name"] == det["name"] and _iou(existing["box"], det["box"]) > 0.45
        ]
        if not overlaps:
            merged.append(det)
            continue
        best_idx = overlaps[0]
        for idx in overlaps[1:]:
            if merged[idx]["conf"] > merged[best_idx]["conf"]:
                best_idx = idx
        if det["conf"] > merged[best_idx]["conf"]:
            merged[best_idx] = det

    # Optionally restrict totals to YOLO when Paddle is disabled; stores rely on PaddleOCR-VL now
    if merged:
        totals = [d for d in merged if d["name"] == "Total"]
        if totals:
            best_total = max(totals, key=lambda d: d["conf"])
            merged = [d for d in merged if d["name"] != "Total" or d is best_total]

    return merged


def _iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)
