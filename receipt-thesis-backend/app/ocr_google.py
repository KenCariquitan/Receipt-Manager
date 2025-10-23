from __future__ import annotations

import os
import json
import asyncio
from typing import Optional

from google.cloud import vision
from google.oauth2 import service_account

GCV_ENABLED = os.getenv("GCP_VISION_ENABLED", "false").lower() == "true"
GCV_CREDENTIALS_PATH = os.getenv("GCP_VISION_CREDENTIALS_PATH", "").strip()
GCV_CREDENTIALS_JSON = os.getenv("GCP_VISION_CREDENTIALS_JSON", "").strip()
_CLIENT: Optional[vision.ImageAnnotatorClient] = None


def _load_credentials():
    if GCV_CREDENTIALS_PATH:
        return service_account.Credentials.from_service_account_file(GCV_CREDENTIALS_PATH)
    if GCV_CREDENTIALS_JSON:
        data = json.loads(GCV_CREDENTIALS_JSON)
        return service_account.Credentials.from_service_account_info(data)
    # fall back to default creds (GOOGLE_APPLICATION_CREDENTIALS)
    return None


def _get_client() -> vision.ImageAnnotatorClient:
    global _CLIENT
    if _CLIENT is None:
        credentials = _load_credentials()
        _CLIENT = vision.ImageAnnotatorClient(credentials=credentials)
    return _CLIENT


async def google_vision_text(img_bytes: bytes, lang_hint: str = "en") -> dict:
    """
    Call Google Cloud Vision's document_text_detection.
    Returns {"ok": bool, "text": str, "error": str|None, "info": dict}
    """
    if not GCV_ENABLED:
        return {"ok": False, "text": "", "error": "disabled", "info": None}

    client = _get_client()
    image = vision.Image(content=img_bytes)
    image_context = {"language_hints": [lang_hint]} if lang_hint else None

    loop = asyncio.get_running_loop()

    def _call():
        return client.document_text_detection(image=image, image_context=image_context)

    try:
        response = await loop.run_in_executor(None, _call)
    except Exception as exc:
        return {"ok": False, "text": "", "error": f"exception:{exc}", "info": None}

    if response.error.message:
        return {"ok": False, "text": "", "error": f"api:{response.error.message}", "info": response._pb.SerializeToString() if hasattr(response, "_pb") else None}  # type: ignore[attr-defined]

    text = response.full_text_annotation.text if response.full_text_annotation else ""

    avg_conf = None
    try:
        confidences = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                confidences.append(block.confidence)
        if confidences:
            avg_conf = float(sum(confidences) / len(confidences))
    except Exception:
        avg_conf = None

    info = {
        "confidence": avg_conf,
        "languages": [prop.language_code for prop in response.text_annotations[0].property.detected_languages] if response.text_annotations else [],
    }
    return {"ok": bool(text.strip()), "text": text or "", "error": None, "info": info}
