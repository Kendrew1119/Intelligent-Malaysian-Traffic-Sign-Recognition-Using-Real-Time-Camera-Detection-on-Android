from __future__ import annotations

import unittest

import numpy as np

from webapp.speed_limit_reader import DEFAULT_MODEL_PATH, SpeedLimitReader


class SpeedLimitReaderTests(unittest.TestCase):
    def test_square_crop_preserves_a_valid_detection(self) -> None:
        image = np.full((120, 200, 3), 220, dtype=np.uint8)
        crop, original_side = SpeedLimitReader._crop_sign(
            image,
            {"x1": 60.0, "y1": 20.0, "x2": 140.0, "y2": 100.0},
        )
        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.shape[0], crop.shape[1])
        self.assertEqual(original_side, 80)

    @unittest.skipUnless(DEFAULT_MODEL_PATH.is_file(), "Local reader model is not installed")
    def test_local_reader_model_loads(self) -> None:
        reader = SpeedLimitReader()
        reader.load()
        self.assertTrue(reader.ready, reader.error)
        self.assertEqual(reader.threshold, 0.85)
        self.assertEqual(reader.status()["values"], ["5", "15", "30", "40", "50", "60", "80"])


if __name__ == "__main__":
    unittest.main()
