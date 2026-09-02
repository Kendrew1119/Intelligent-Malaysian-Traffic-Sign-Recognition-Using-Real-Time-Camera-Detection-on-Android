from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np


SPEED_LIMIT_CLASSES = {
    "speed-limit-5",
    "speed-limit-15",
    "speed-limit-30",
    "speed-limit-40",
    "speed-limit-50",
    "speed-limit-60",
    "speed-limit-80",
}
PLAUSIBLE_SPEEDS = frozenset(range(5, 135, 5))
OCR_OVERRIDE_CONFIDENCE = 0.95
OCR_CONFIRM_CONFIDENCE = 0.70
MINIMUM_CROP_SIDE = 14
INNER_CROP_RATIO = 0.12
MINIMUM_RECOGNITION_HEIGHT = 96


class SpeedLimitOCR:
    """Read digits from a YOLO-proposed speed-limit crop.

    OCR is advisory. A failed or low-confidence OCR result never suppresses the
    detector result. A different value replaces YOLO's numeric class only when
    it is a plausible road speed and passes the strict confidence threshold.
    """

    def __init__(
        self,
        engine: Callable[..., Any] | None = None,
        override_confidence: float = OCR_OVERRIDE_CONFIDENCE,
    ) -> None:
        self.engine = engine
        self.override_confidence = override_confidence
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.engine is not None:
            return
        try:
            from rapidocr import RapidOCR

            self.engine = RapidOCR(
                params={
                    "Global.use_det": False,
                    "Global.use_cls": False,
                    "Global.use_rec": True,
                }
            )
            self.error = None
        except Exception as exc:  # OCR must never prevent YOLO startup.
            self.engine = None
            self.error = str(exc)

    @property
    def ready(self) -> bool:
        return self.engine is not None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.ready,
            "engine": "RapidOCR" if self.ready else None,
            "confirm_confidence": OCR_CONFIRM_CONFIDENCE,
            "override_confidence": self.override_confidence,
            "plausible_values": sorted(PLAUSIBLE_SPEEDS),
            "error": self.error,
        }

    @staticmethod
    def _prepare_crop(
        image: np.ndarray,
        bbox: dict[str, float],
    ) -> tuple[np.ndarray | None, int]:
        image_height, image_width = image.shape[:2]
        x1 = max(0, min(image_width, int(np.floor(bbox["x1"]))))
        y1 = max(0, min(image_height, int(np.floor(bbox["y1"]))))
        x2 = max(0, min(image_width, int(np.ceil(bbox["x2"]))))
        y2 = max(0, min(image_height, int(np.ceil(bbox["y2"]))))
        width = x2 - x1
        height = y2 - y1
        original_side = min(width, height)
        if original_side < MINIMUM_CROP_SIDE:
            return None, original_side

        crop = image[y1:y2, x1:x2]
        inset_x = max(1, int(round(crop.shape[1] * INNER_CROP_RATIO)))
        inset_y = max(1, int(round(crop.shape[0] * INNER_CROP_RATIO)))
        inner = crop[inset_y : crop.shape[0] - inset_y, inset_x : crop.shape[1] - inset_x]
        if inner.size == 0:
            return None, original_side

        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < MINIMUM_RECOGNITION_HEIGHT:
            scale = MINIMUM_RECOGNITION_HEIGHT / gray.shape[0]
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        prepared = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return prepared, original_side

    @staticmethod
    def _detector_value(class_name: str) -> int | None:
        try:
            return int(class_name.rsplit("-", 1)[-1])
        except (TypeError, ValueError):
            return None

    def read(
        self,
        image: np.ndarray,
        bbox: dict[str, float],
        detector_class_name: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if detector_class_name not in SPEED_LIMIT_CLASSES:
            return {"status": "not-speed-limit", "applied": False, "ocr_ms": 0.0}

        if self.engine is None:
            self.load()
        if self.engine is None:
            return {
                "status": "unavailable",
                "applied": False,
                "ocr_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": self.error,
            }

        crop, original_side = self._prepare_crop(image, bbox)
        if crop is None:
            return {
                "status": "too-small",
                "applied": False,
                "crop_side": original_side,
                "ocr_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        try:
            with self._lock:
                result = self.engine(
                    crop,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
        except Exception as exc:
            self.error = str(exc)
            return {
                "status": "error",
                "applied": False,
                "crop_side": original_side,
                "ocr_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": self.error,
            }

        texts = tuple(getattr(result, "txts", ()) or ())
        scores = tuple(getattr(result, "scores", ()) or ())
        raw_text = str(texts[0]).strip() if texts else ""
        confidence = float(scores[0]) if scores else 0.0
        digit_text = re.sub(r"[^0-9]", "", raw_text)
        value = int(digit_text) if digit_text and len(digit_text) <= 3 else None
        detector_value = self._detector_value(detector_class_name)
        plausible = value in PLAUSIBLE_SPEEDS
        agrees = bool(
            plausible
            and value == detector_value
            and confidence >= OCR_CONFIRM_CONFIDENCE
        )
        override = bool(
            plausible
            and value != detector_value
            and confidence >= self.override_confidence
        )

        if agrees:
            status = "confirmed"
        elif override:
            status = "override"
        elif not raw_text:
            status = "no-text"
        elif not plausible:
            status = "not-plausible"
        else:
            status = "low-confidence"

        return {
            "status": status,
            "raw_text": raw_text,
            "value": value if plausible else None,
            "confidence": round(confidence, 4),
            "applied": agrees or override,
            "overrode_yolo": override,
            "crop_side": original_side,
            "ocr_ms": round((time.perf_counter() - started) * 1000, 1),
        }
