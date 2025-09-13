from __future__ import annotations
import cv2, numpy as np, pytesseract
from PIL import Image
import io

# If needed on Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

DEF_LANG = "eng"


def _unsharp(gray):
    blur = cv2.GaussianBlur(gray, (0,0), 3)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

def auto_deskew(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 10: return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = auto_deskew(gray)
    gray = _unsharp(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 9)
    th = cv2.dilate(th, np.ones((1,1), np.uint8), iterations=1)
    return th

def _decode_bytes_to_bgr(data: bytes) -> np.ndarray | None:
    # Try OpenCV imdecode
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # Fallback to PIL → numpy
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def ocr_image_bytes(data: bytes, lang: str = DEF_LANG) -> dict:
    img = _decode_bytes_to_bgr(data)
    if img is None:
        raise ValueError("Cannot decode image bytes (unsupported/invalid format).")
    prep = preprocess(img)

    def _ocr_pass(psm: int, allowlist: str | None = None):
        cfg = f"--psm {psm}"
        if allowlist:
            cfg += f" -c tessedit_char_whitelist={allowlist}"
        txt = pytesseract.image_to_string(prep, lang=lang, config=cfg)
        df = pytesseract.image_to_data(prep, lang=lang, config=cfg, output_type=pytesseract.Output.DATAFRAME)
        df = df[df.conf != -1] if 'conf' in df else df
        mean_conf = float(df.conf.mean()) if len(df) else float("nan")
        return txt, mean_conf

    # Try a few PSMs and keep the best by mean confidence
    tries = []
    for psm in (6, 4, 11, 3):
        try:
            t, c = _ocr_pass(psm)
            tries.append((c, t, psm))
        except Exception:
            pass

    if tries:
        c, t, used_psm = max(tries, key=lambda x: x[0])
    else:
        # last resort
        t, c = pytesseract.image_to_string(prep, lang=lang), float("nan")
        used_psm = -1

    # Amount-friendly pass
    amt_text, _ = _ocr_pass(6, allowlist="0123456789.,₱PHPPhp ")

    return {"text": t, "mean_conf": c, "w": prep.shape[1], "h": prep.shape[0], "psm": used_psm, "amount_pass": amt_text}

def ocr_image_path(path: str, lang: str = DEF_LANG) -> dict:
    # Read bytes to avoid Windows/unicode path quirks
    with open(path, "rb") as f:
        data = f.read()
    return ocr_image_bytes(data, lang=lang)

def ocr_crop(img_bgr, box, psm=7, allowlist=None, lang=DEF_LANG):
    """
    OCR only the given bounding box. Good for 'store_header', 'total_value', 'date_box'.
    psm 7 = single text line (good for totals/store).
    """
    x1, y1, x2, y2 = box
    crop = img_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return ""

    # Upscale small regions to help OCR
    crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Light preprocessing
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 9)

    cfg = f"--psm {psm}"
    if allowlist:
        cfg += f" -c tessedit_char_whitelist={allowlist}"
    txt = pytesseract.image_to_string(th, lang=lang, config=cfg)
    return txt.strip()