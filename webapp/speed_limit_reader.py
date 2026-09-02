"""Conservative second-stage reader for numbered speed-limit detections."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "speed_limit_reader.pt"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "speed_limit_reader_manifest.json"
SPEED_LIMIT_CLASSES = {
    "speed-limit-5",
    "speed-limit-15",
    "speed-limit-30",
    "speed-limit-40",
    "speed-limit-50",
    "speed-limit-60",
    "speed-limit-80",
}


class SpeedLimitReader:
    """Read one of the supported numeric values from a YOLO sign crop."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.model: torch.jit.ScriptModule | None = None
        self.manifest: dict[str, Any] = {}
        self.error: str | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
            values = self.manifest.get("values", [])
            if values != ["5", "15", "30", "40", "50", "60", "80"]:
                raise ValueError("The speed-limit reader has an unexpected class order.")
            self.error = None
        except Exception as exc:
            self.model = None
            self.error = str(exc)

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        return float(self.manifest.get("confidence_threshold", 0.85))

    @property
    def minimum_crop_side(self) -> int:
        return int(self.manifest.get("minimum_crop_side", 40))

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "values": self.manifest.get("values", []),
            "confidence_threshold": self.threshold,
            "minimum_crop_side": self.minimum_crop_side,
            "error": self.error,
        }

    def read(self, image: np.ndarray, bbox: dict[str, float]) -> dict[str, Any]:
        if self.model is None:
            self.load()
        if self.model is None:
            return {"status": "unavailable", "confidence": 0.0, "value": None}

        started = time.perf_counter()
        crop, original_side = self._crop_sign(image, bbox)
        if crop is None or original_side < self.minimum_crop_side:
            return {
                "status": "too-small",
                "confidence": 0.0,
                "value": None,
                "minimum_crop_side": self.minimum_crop_side,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        image_size = int(self.manifest.get("image_size", 96))
        resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensor = tensor.sub(0.5).div(0.5).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        confidence, class_index = probabilities.max(dim=0)
        confidence_value = float(confidence.item())
        value = str(self.manifest["values"][int(class_index.item())])
        return {
            "status": "confirmed" if confidence_value >= self.threshold else "uncertain",
            "value": value,
            "confidence": round(confidence_value, 4),
            "threshold": self.threshold,
            "crop_side": original_side,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    @staticmethod
    def _crop_sign(
        image: np.ndarray,
        bbox: dict[str, float],
        padding: float = 0.12,
    ) -> tuple[np.ndarray | None, int]:
        height, width = image.shape[:2]
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        original_side = int(round(min(box_width, box_height)))
        if original_side <= 0:
            return None, 0
        pad_x = max(3.0, box_width * padding)
        pad_y = max(3.0, box_height * padding)
        left = max(0, int(np.floor(x1 - pad_x)))
        top = max(0, int(np.floor(y1 - pad_y)))
        right = min(width, int(np.ceil(x2 + pad_x)))
        bottom = min(height, int(np.ceil(y2 + pad_y)))
        if right - left < 8 or bottom - top < 8:
            return None, original_side

        crop = image[top:bottom, left:right]
        crop_height, crop_width = crop.shape[:2]
        side = max(crop_width, crop_height)
        canvas = np.full((side, side, 3), 238, dtype=np.uint8)
        offset_x = (side - crop_width) // 2
        offset_y = (side - crop_height) // 2
        canvas[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width] = crop
        return canvas, original_side
