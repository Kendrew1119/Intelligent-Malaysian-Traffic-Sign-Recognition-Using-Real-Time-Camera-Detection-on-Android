from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from webapp.speed_limit_ocr import SpeedLimitOCR


class FakeOCREngine:
    def __init__(self, text: str, confidence: float) -> None:
        self.text = text
        self.confidence = confidence
        self.last_crop = None

    def __call__(self, crop, **_kwargs):
        self.last_crop = crop
        return SimpleNamespace(txts=(self.text,), scores=(self.confidence,))


class SpeedLimitOCRTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((160, 200, 3), 230, dtype=np.uint8)
        self.bbox = {"x1": 40.0, "y1": 20.0, "x2": 160.0, "y2": 140.0}

    def test_high_confidence_additional_value_overrides_yolo(self) -> None:
        engine = FakeOCREngine("70", 0.99)
        reader = SpeedLimitOCR(engine=engine)
        result = reader.read(self.image, self.bbox, "speed-limit-80")

        self.assertEqual(result["status"], "override")
        self.assertEqual(result["value"], 70)
        self.assertTrue(result["applied"])
        self.assertTrue(result["overrode_yolo"])
        self.assertIsNotNone(engine.last_crop)
        self.assertEqual(engine.last_crop.ndim, 3)

    def test_low_confidence_different_value_keeps_yolo(self) -> None:
        reader = SpeedLimitOCR(engine=FakeOCREngine("70", 0.80))
        result = reader.read(self.image, self.bbox, "speed-limit-80")

        self.assertEqual(result["status"], "low-confidence")
        self.assertFalse(result["applied"])
        self.assertFalse(result["overrode_yolo"])

    def test_matching_value_confirms_without_override(self) -> None:
        reader = SpeedLimitOCR(engine=FakeOCREngine("80", 0.70))
        result = reader.read(self.image, self.bbox, "speed-limit-80")

        self.assertEqual(result["status"], "confirmed")
        self.assertTrue(result["applied"])
        self.assertFalse(result["overrode_yolo"])

    def test_implausible_number_is_rejected(self) -> None:
        reader = SpeedLimitOCR(engine=FakeOCREngine("999", 1.0))
        result = reader.read(self.image, self.bbox, "speed-limit-80")

        self.assertEqual(result["status"], "not-plausible")
        self.assertIsNone(result["value"])
        self.assertFalse(result["applied"])


if __name__ == "__main__":
    unittest.main()
