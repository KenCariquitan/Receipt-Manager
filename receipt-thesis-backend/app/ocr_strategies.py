from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from .ocr import ocr_image_bytes, DEF_LANG
from .ocr_space import ocr_space_bytes
from .ocr_google import google_vision_text
from .ocr_paddle_vl import paddle_vl_text


@dataclass
class OCRResult:
    """Normalized response returned by every OCR strategy."""

    name: str
    payload: Dict[str, Any]


class OCRStrategy(ABC):
    """Interface that every OCR provider implements."""

    name: str

    @abstractmethod
    async def recognize(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        lang: str = DEF_LANG,
    ) -> OCRResult:
        """Run OCR and return a normalized payload."""


class TesseractStrategy(OCRStrategy):
    name = "tesseract"

    async def recognize(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        lang: str = DEF_LANG,
    ) -> OCRResult:
        loop = asyncio.get_running_loop()

        def _run():
            return ocr_image_bytes(image_bytes, lang=lang)

        payload = await loop.run_in_executor(None, _run)
        payload.setdefault("filename", filename)
        return OCRResult(self.name, payload)


class OCRSpaceStrategy(OCRStrategy):
    name = "ocr_space"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    async def recognize(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        lang: str = DEF_LANG,
    ) -> OCRResult:
        if not self.enabled:
            return OCRResult(
                self.name,
                {"ok": False, "text": "", "raw": None, "error": "disabled", "http": None},
            )
        payload = await ocr_space_bytes(image_bytes, filename=filename, lang=lang)
        return OCRResult(self.name, payload)


class GoogleVisionStrategy(OCRStrategy):
    name = "google_vision"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    async def recognize(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        lang: str = DEF_LANG,
    ) -> OCRResult:
        if not self.enabled:
            return OCRResult(
                self.name,
                {"ok": False, "text": "", "error": "disabled", "info": None},
            )
        payload = await google_vision_text(image_bytes, lang_hint=lang)
        return OCRResult(self.name, payload)


class PaddleVLStrategy(OCRStrategy):
    name = "paddle_vl"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    async def recognize(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        lang: str = DEF_LANG,
    ) -> OCRResult:
        if not self.enabled:
            return OCRResult(
                self.name,
                {"ok": False, "text": "", "error": "disabled"},
            )
        loop = asyncio.get_running_loop()

        def _run():
            return paddle_vl_text(image_bytes)

        payload = await loop.run_in_executor(None, _run)
        payload.setdefault("filename", filename)
        return OCRResult(self.name, payload)


class OCRContext:
    """Simple context to look up and execute OCR strategies by name."""

    def __init__(self, strategies: Dict[str, OCRStrategy]) -> None:
        self._strategies = strategies

    def has(self, name: str) -> bool:
        return name in self._strategies

    async def run(
        self,
        name: str,
        *,
        image_bytes: bytes,
        filename: str,
        lang: str = DEF_LANG,
    ) -> OCRResult:
        strategy = self._strategies[name]
        return await strategy.recognize(image_bytes=image_bytes, filename=filename, lang=lang)
