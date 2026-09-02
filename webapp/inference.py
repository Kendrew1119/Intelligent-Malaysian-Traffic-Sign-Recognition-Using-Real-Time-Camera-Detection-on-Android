from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from webapp.speed_limit_ocr import SPEED_LIMIT_CLASSES, SpeedLimitOCR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_openvino_model"
MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"


class DetectorUnavailableError(RuntimeError):
    """Raised when the configured detector cannot be loaded."""


class SignDetector:
    """Thread-safe wrapper around the exported YOLO26s OpenVINO model."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        image_size: int = 640,
        speed_limit_ocr: SpeedLimitOCR | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.image_size = image_size
        self.model: YOLO | None = None
        self.class_names: dict[int, str] = {}
        self.manifest = self._read_manifest()
        self.speed_limit_ocr = speed_limit_ocr or SpeedLimitOCR()
        self._lock = threading.Lock()

    def _read_manifest(self) -> dict[str, Any]:
        if not MANIFEST_PATH.is_file():
            return {}
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def load(self, warmup: bool = True) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise DetectorUnavailableError(
                f"Model export was not found at {self.model_path}"
            )

        try:
            model = YOLO(str(self.model_path), task="detect")
            names = model.names
            if isinstance(names, list):
                names = dict(enumerate(names))
            self.class_names = {int(key): value for key, value in names.items()}
            expected_classes = int(self.manifest.get("classes", 63))
            if len(self.class_names) != expected_classes:
                raise DetectorUnavailableError(
                    f"Expected {expected_classes} classes, found {len(self.class_names)}"
                )
            self.model = model
            self.speed_limit_ocr.load()
            if warmup:
                blank_frame = np.full((640, 640, 3), 245, dtype=np.uint8)
                self.predict(blank_frame, confidence=0.20)
        except DetectorUnavailableError:
            raise
        except Exception as exc:
            self.model = None
            raise DetectorUnavailableError(f"Could not load the model: {exc}") from exc

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, image: np.ndarray, confidence: float = 0.20) -> dict[str, Any]:
        if self.model is None:
            self.load(warmup=False)
        assert self.model is not None

        started = time.perf_counter()
        with self._lock:
            results = self.model.predict(
                source=image,
                imgsz=self.image_size,
                conf=confidence,
                verbose=False,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = results[0]
        detections: list[dict[str, Any]] = []
        ocr_ms = 0.0
        class_ids_by_name = {name: class_id for class_id, name in self.class_names.items()}

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = (round(float(value), 2) for value in box)
                class_name = self.class_names[int(class_id)]
                detection = {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": round(float(score), 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
                if class_name in SPEED_LIMIT_CLASSES:
                    reading = self.speed_limit_ocr.read(
                        image,
                        detection["bbox"],
                        class_name,
                    )
                    ocr_ms += float(reading.get("ocr_ms", 0.0))
                    detection["speed_limit_ocr"] = reading
                    detection["classification_source"] = (
                        "yolo+ocr" if reading.get("status") == "confirmed" else "yolo"
                    )
                    if reading.get("overrode_yolo"):
                        ocr_class_name = f"speed-limit-{reading['value']}"
                        detection["detector_class_id"] = detection["class_id"]
                        detection["detector_class_name"] = detection["class_name"]
                        detection["class_name"] = ocr_class_name
                        detection["class_id"] = class_ids_by_name.get(
                            ocr_class_name,
                            detection["class_id"],
                        )
                        detection["classification_source"] = "ocr"
                detections.append(detection)

        return {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "inference_ms": round(elapsed_ms, 1),
            "ocr_ms": round(ocr_ms, 1),
            "detections": detections,
        }


def decode_image(payload: bytes) -> np.ndarray:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The selected file is not a readable image.")
    return image
