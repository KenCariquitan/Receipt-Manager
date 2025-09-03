from __future__ import annotations
import cv2, numpy as np, pytesseract

# If needed on Windows, set the Tesseract path like this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

DEF_LANG = "eng"

def auto_deskew(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = auto_deskew(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 9)
    return th

def ocr_image_path(path: str, lang: str = DEF_LANG) -> dict:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    prep = preprocess(img)
    text = pytesseract.image_to_string(prep, lang=lang)
    tsv = pytesseract.image_to_data(prep, lang=lang, output_type=pytesseract.Output.DATAFRAME)
    conf = float(tsv[tsv.conf != -1].conf.mean()) if 'conf' in tsv else float("nan")
    return {"path": path, "text": text, "mean_conf": conf, "w": prep.shape[1], "h": prep.shape[0]}
