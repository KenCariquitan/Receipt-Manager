from __future__ import annotations
import os, io, math
import httpx
from PIL import Image

OCR_URL = os.getenv("OCR_SPACE_URL", "https://api.ocr.space/parse/image")
OCR_KEY = os.getenv("OCR_SPACE_API_KEY", "")
OCR_ENABLED = os.getenv("OCR_SPACE_ENABLED", "false").lower() == "true"

MAX_BYTES = 1_200_000  # ~1.2MB cap for OCR.space (free plan tolerance)

def _maybe_downscale(img_bytes: bytes) -> bytes:
    if len(img_bytes) <= MAX_BYTES:
        return img_bytes
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        scale = math.sqrt(MAX_BYTES / len(img_bytes))
        new_w = max(600, int(im.width * scale))
        new_h = max(600, int(im.height * scale))
        im = im.resize((new_w, new_h))
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=85, optimize=True)
        out_bytes = out.getvalue()
        return out_bytes if len(out_bytes) < len(img_bytes) else img_bytes
    except Exception:
        return img_bytes

async def ocr_space_bytes(img_bytes: bytes, filename: str = "receipt.jpg", lang: str = "eng") -> dict:
    if not OCR_ENABLED or not OCR_KEY:
        return {"ok": False, "text": "", "raw": None, "error": "disabled_or_no_key", "http": None}

    send_bytes = _maybe_downscale(img_bytes)

    data = {"language": lang, "isOverlayRequired": False, "OCREngine": 2, "scale": True, "isTable": False}
    headers = {"apikey": OCR_KEY}
    files = {"file": (filename, send_bytes, "application/octet-stream")}

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(OCR_URL, data=data, headers=headers, files=files)
            http_code = r.status_code
            j = r.json()
    except Exception as e:
        return {"ok": False, "text": "", "raw": None, "error": f"network:{e}", "http": None}

    text = ""
    if isinstance(j, dict) and j.get("ParsedResults"):
        text = "\n".join(pr.get("ParsedText", "") for pr in j["ParsedResults"]).strip()

    err = None
    if isinstance(j, dict) and j.get("IsErroredOnProcessing"):
        err = f"api:{j.get('ErrorMessage') or j.get('ErrorDetails') or 'unknown'}"

    ok = bool(text) and not j.get("IsErroredOnProcessing", False)
    return {"ok": ok, "text": text or "", "raw": j, "error": err, "http": http_code}
