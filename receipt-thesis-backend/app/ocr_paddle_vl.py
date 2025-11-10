from __future__ import annotations

import io
from typing import Any, Dict, List

from PIL import Image

_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "paddleocr is not installed. Install PaddleOCR-VL to enable this strategy."
            ) from exc
        _PIPELINE = PaddleOCRVL()
    return _PIPELINE


def _collect_texts(obj: Any, out: List[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "text" and isinstance(v, str):
                out.append(v)
            else:
                _collect_texts(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_texts(item, out)


def paddle_vl_text(image_bytes: bytes, prompt: str | None = None) -> Dict[str, Any]:
    """
    Run PaddleOCR-VL locally and return a text blob compatible with parse_fields.
    """
    try:
        pipeline = _load_pipeline()
    except Exception as exc:
        return {"ok": False, "text": "", "error": f"load:{exc}"}

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return {"ok": False, "text": "", "error": f"decode:{exc}"}

    try:
        if prompt:
            results = pipeline.predict([{"image": img, "prompt": prompt}])
        else:
            results = pipeline.predict(img)
    except Exception as exc:
        return {"ok": False, "text": "", "error": f"infer:{exc}"}

    texts: List[str] = []
    for res in results:
        payload = res.to_dict() if hasattr(res, "to_dict") else {}
        _collect_texts(payload, texts)

    merged = "\n".join(texts).strip()
    return {"ok": bool(merged), "text": merged, "error": None}
