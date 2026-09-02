from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from fastapi.testclient import TestClient

from webapp.main import create_app
from webapp.hard_cases import HardCaseStore
from webapp.sign_catalog import SIGN_DETAILS


class FakeOCR:
    def status(self) -> dict:
        return {
            "enabled": True,
            "engine": "RapidOCR",
            "override_confidence": 0.95,
            "plausible_values": list(range(5, 135, 5)),
            "error": None,
        }


class FakeDetector:
    ready = False
    manifest = {"model": "YOLO26s"}
    image_size = 640
    class_names = {index: name for index, name in enumerate(SIGN_DETAILS)}
    speed_limit_ocr = FakeOCR()

    def load(self) -> None:
        self.ready = True

    def predict(self, image, confidence: float):
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "inference_ms": 12.3,
            "detections": [
                {
                    "class_id": 30,
                    "class_name": "no-uturn",
                    "confidence": 0.91,
                    "bbox": {"x1": 2.0, "y1": 3.0, "x2": 20.0, "y2": 21.0},
                }
            ],
        }


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = TemporaryDirectory()
        cls.hard_case_store = HardCaseStore(cls.temp_directory.name)
        cls.client_context = TestClient(create_app(FakeDetector(), cls.hard_case_store))
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        cls.temp_directory.cleanup()

    def test_home_and_health(self) -> None:
        home = self.client.get("/")
        health = self.client.get("/api/health")
        self.assertEqual(home.status_code, 200)
        self.assertIn("MYSignVoice", home.text)
        self.assertTrue(health.json()["ready"])
        self.assertEqual(health.json()["classes"], 63)
        self.assertEqual(health.json()["speed_limit_mode"]["mode"], "yolo-plus-ocr")
        self.assertFalse(health.json()["speed_limit_mode"]["reader_enabled"])
        self.assertTrue(health.json()["speed_limit_mode"]["ocr"]["enabled"])
        signs = self.client.get("/api/signs")
        self.assertEqual(signs.status_code, 200)
        self.assertEqual(signs.json()["count"], 63)
        no_uturn = next(sign for sign in signs.json()["signs"] if sign["class_name"] == "no-uturn")
        self.assertEqual(no_uturn["speech"], "No U-turn ahead.")

    def test_detect_image(self) -> None:
        success, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(success)
        response = self.client.post(
            "/api/detect",
            files={"image": ("sign.png", encoded.tobytes(), "image/png")},
            data={"confidence": "0.20"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detections"][0]["class_name"], "no-uturn")

    def test_rejects_non_image(self) -> None:
        response = self.client.post(
            "/api/detect",
            files={"image": ("notes.txt", b"not an image", "text/plain")},
            data={"confidence": "0.20"},
        )
        self.assertEqual(response.status_code, 415)

    def test_saves_difficult_frame_with_metadata(self) -> None:
        success, encoded = cv2.imencode(".jpg", np.zeros((12, 12, 3), dtype=np.uint8))
        self.assertTrue(success)
        response = self.client.post(
            "/api/hard-cases",
            files={"image": ("camera-hard-case.jpg", encoded.tobytes(), "image/jpeg")},
            data={
                "source": "camera",
                "issue_type": "missed",
                "expected_class": "no-uturn",
                "predicted_classes": "[]",
                "notes": "far away",
                "confidence": "0.20",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["saved"])
        self.assertEqual(self.hard_case_store.count(), 1)
        self.assertEqual(len(list(self.hard_case_store.image_dir.glob("*.jpg"))), 1)


if __name__ == "__main__":
    unittest.main()
