"""Evaluate YOLO-only and conservative two-stage speed-limit reading."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.prepare_speed_limit_reader import (
    SPEED_CLASS_TO_VALUE,
    find_image_entry,
    read_config,
)
from webapp.inference import DEFAULT_MODEL_PATH, SignDetector
from webapp.speed_limit_reader import SPEED_LIMIT_CLASSES


DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "speed_limit_reader_report" / "hybrid_test.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.20)
    return parser.parse_args()


def pixel_box(row: tuple[float, float, float, float], width: int, height: int) -> dict[str, float]:
    center_x, center_y, box_width, box_height = row
    return {
        "x1": (center_x - box_width / 2) * width,
        "y1": (center_y - box_height / 2) * height,
        "x2": (center_x + box_width / 2) * width,
        "y2": (center_y + box_height / 2) * height,
    }


def iou(first: dict[str, float], second: dict[str, float]) -> float:
    left = max(first["x1"], second["x1"])
    top = max(first["y1"], second["y1"])
    right = min(first["x2"], second["x2"])
    bottom = min(first["y2"], second["y2"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first["x2"] - first["x1"]) * max(0.0, first["y2"] - first["y1"])
    second_area = max(0.0, second["x2"] - second["x1"]) * max(0.0, second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def main() -> None:
    args = parse_args()
    detector = SignDetector(args.model.resolve())
    detector.load(warmup=True)
    rows: list[dict[str, object]] = []
    image_results: dict[str, dict] = {}

    with zipfile.ZipFile(args.zip_path.resolve()) as archive:
        config = read_config(archive)
        names: list[str] = config["names"]
        entries = set(archive.namelist())
        test_labels = sorted(
            name for name in entries if name.startswith("test/labels/") and name.endswith(".txt")
        )
        for label_entry in test_labels:
            speed_labels = []
            for line in archive.read(label_entry).decode("utf-8").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                class_id = int(float(fields[0]))
                class_name = names[class_id]
                if class_name in SPEED_CLASS_TO_VALUE:
                    speed_labels.append((class_name, tuple(map(float, fields[1:5]))))
            if not speed_labels:
                continue
            image_entry = find_image_entry(entries, label_entry)
            if image_entry is None:
                continue
            encoded = np.frombuffer(archive.read(image_entry), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                continue
            started = time.perf_counter()
            prediction = detector.predict(image, confidence=args.confidence)
            prediction["pipeline_ms"] = round((time.perf_counter() - started) * 1000, 1)
            image_results[image_entry] = prediction
            for expected_class, yolo_box in speed_labels:
                expected_box = pixel_box(yolo_box, image.shape[1], image.shape[0])
                candidates = [
                    (iou(expected_box, detection["bbox"]), detection)
                    for detection in prediction["detections"]
                ]
                overlap, matched = max(candidates, default=(0.0, None), key=lambda item: item[0])
                if overlap < 0.30:
                    matched = None
                detector_class = matched.get("detector_class_name", matched["class_name"]) if matched else None
                reading = matched.get("speed_limit_reading", {}) if matched else {}
                speed_family_detected = bool(matched and detector_class in SPEED_LIMIT_CLASSES)
                rows.append(
                    {
                        "source_image": image_entry,
                        "expected_class": expected_class,
                        "matched_iou": round(overlap, 4),
                        "speed_family_detected": speed_family_detected,
                        "detector_class": detector_class,
                        "detector_exact": detector_class == expected_class,
                        "detector_confidence": matched["confidence"] if matched else None,
                        "reader_status": reading.get("status"),
                        "reader_value": reading.get("value"),
                        "reader_confidence": reading.get("confidence"),
                        "hybrid_spoken": reading.get("status") == "confirmed",
                        "hybrid_exact": (
                            reading.get("status") == "confirmed"
                            and f"speed-limit-{reading.get('value')}" == expected_class
                        ),
                        "yolo_inference_ms": prediction["inference_ms"],
                        "full_pipeline_ms": prediction["pipeline_ms"],
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    speed_detected = sum(bool(row["speed_family_detected"]) for row in rows)
    detector_exact = sum(bool(row["detector_exact"]) for row in rows)
    hybrid_spoken = sum(bool(row["hybrid_spoken"]) for row in rows)
    hybrid_exact = sum(bool(row["hybrid_exact"]) for row in rows)
    summary = {
        "test_speed_limit_instances": total,
        "speed_family_recall": speed_detected / total if total else 0.0,
        "detector_exact_accuracy_all_instances": detector_exact / total if total else 0.0,
        "hybrid_spoken_coverage": hybrid_spoken / total if total else 0.0,
        "hybrid_spoken_accuracy": hybrid_exact / hybrid_spoken if hybrid_spoken else 0.0,
        "unique_test_images": len(image_results),
        "mean_full_pipeline_ms": (
            sum(float(result["pipeline_ms"]) for result in image_results.values()) / len(image_results)
            if image_results else 0.0
        ),
        "confidence_threshold": args.confidence,
        "output_csv": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
