from __future__ import annotations

import io
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

DEF_LANG = "eng"
AMOUNT_RE = re.compile(r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))")


def _unsharp(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def auto_deskew(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = auto_deskew(gray)
    gray = _unsharp(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    th = cv2.dilate(th, np.ones((1, 1), np.uint8), iterations=1)
    return th


def _decode_bytes_to_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
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
        if "conf" in df:
            df = df[df.conf != -1]
        mean_conf = float(df.conf.mean()) if len(df) else float("nan")
        return txt, mean_conf, df

    best_df = None
    tries = []
    for psm in (6, 4, 11, 3):
        try:
            t, c, df = _ocr_pass(psm)
            tries.append((c, t, psm, df))
        except Exception:
            continue

    if tries:
        c, t, used_psm, best_df = max(tries, key=lambda x: x[0])
    else:
        t, c, best_df = pytesseract.image_to_string(prep, lang=lang), float("nan"), None
        used_psm = -1
        try:
            _, _, best_df = _ocr_pass(6)
        except Exception:
            best_df = None

    words = []
    if best_df is not None and len(best_df):
        for _, row in best_df.iterrows():
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            try:
                words.append(
                    {
                        "text": text,
                        "conf": float(row.get("conf", 0.0)),
                        "left": int(row.get("left", 0)),
                        "top": int(row.get("top", 0)),
                        "width": int(row.get("width", 0)),
                        "height": int(row.get("height", 0)),
                        "block_num": int(row.get("block_num", 0)),
                        "par_num": int(row.get("par_num", 0)),
                        "line_num": int(row.get("line_num", 0)),
                        "word_num": int(row.get("word_num", 0)),
                    }
                )
            except Exception:
                continue

    amt_text, _, _ = _ocr_pass(6, allowlist="0123456789.,?PHPPhp ")

    return {
        "text": t,
        "mean_conf": c,
        "w": prep.shape[1],
        "h": prep.shape[0],
        "psm": used_psm,
        "amount_pass": amt_text,
        "words": words,
    }


def ocr_image_path(path: str, lang: str = DEF_LANG) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    return ocr_image_bytes(data, lang=lang)


def ocr_crop(img_bgr, box, psm=7, allowlist=None, lang=DEF_LANG):
    x1, y1, x2, y2 = box
    crop = img_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return ""

    crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    cfg = f"--psm {psm}"
    if allowlist:
        cfg += f" -c tessedit_char_whitelist={allowlist}"
    txt = pytesseract.image_to_string(th, lang=lang, config=cfg)
    return txt.strip()


def ocr_amount_from_crop(img_bgr, box, lang: str = DEF_LANG):
    """
    Run multiple OCR passes on the detected total region and extract the most confident amount.
    Returns (value: Optional[float], raw_text: str, tried_texts: list[str])
    """
    tried: list[str] = []
    best_val: float | None = None
    best_text: str = ""

    for psm in (7, 6, 5, 11):
        for allow in ("0123456789.,₱PHPphp ", None):
            txt = ocr_crop(img_bgr, box, psm=psm, allowlist=allow, lang=lang)
            if not txt:
                continue
            cleaned = (
                txt.replace("PHP", "")
                .replace("Php", "")
                .replace("php", "")
                .replace("₱", "")
                .strip()
            )
            tried.append(cleaned)
            for match in AMOUNT_RE.finditer(cleaned):
                try:
                    val = float(match.group(1).replace(",", ""))
                except Exception:
                    continue
                candidate_len = len(match.group(1))
                current_len = len(f"{best_val}") if best_val is not None else 0
                if (
                    best_val is None
                    or candidate_len > current_len
                    or val > (best_val or 0)
                ):
                    best_val = val
                    best_text = cleaned

    return best_val, best_text, tried
