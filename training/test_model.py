"""Run a YOLO26 image smoke test and save annotated images plus detections.csv.

This command checks inference behavior; it does not calculate accuracy or mAP.
Use evaluate.py with a labeled validation/test split for quantitative metrics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "image_smoke_test"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="An image file or a directory of images.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained YOLO26 weights or export.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.4, help="Minimum detection confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cpu or 0. Omit for Ultralytics auto-selection.",
    )
    return parser.parse_args()


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def detection_rows(result: object, output_name: str) -> Iterable[list[object]]:
    boxes = result.boxes
    inference_ms = float(result.speed.get("inference", 0.0))
    if boxes is None or len(boxes) == 0:
        yield [str(result.path), output_name, "", "", "", "", "", "", "", inference_ms]
        return

    for xyxy, confidence, class_id in zip(
        boxes.xyxy.cpu().tolist(),
        boxes.conf.cpu().tolist(),
        boxes.cls.cpu().tolist(),
    ):
        numeric_class_id = int(class_id)
        yield [
            str(result.path),
            output_name,
            numeric_class_id,
            result.names[numeric_class_id],
            float(confidence),
            *[float(value) for value in xyxy],
            inference_ms,
        ]


def main() -> None:
    args = parse_args()
    images = find_images(args.source)
    if not images:
        raise SystemExit(f"No supported images found at: {args.source}")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be greater than zero")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model, task="detect")
    predict_options = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_options["device"] = args.device

    csv_path = args.output / "detections.csv"
    detection_count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["source", "annotated_image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "inference_ms"]
        )

        for index, image_path in enumerate(images, start=1):
            result = model.predict(source=str(image_path), **predict_options)[0]
            source_name = image_path.stem or "image"
            output_name = f"{index:04d}_{source_name}.jpg"
            output_path = args.output / output_name
            if not cv2.imwrite(str(output_path), result.plot()):
                raise RuntimeError(f"Could not write annotated image: {output_path}")

            rows = list(detection_rows(result, output_name))
            detection_count += sum(1 for row in rows if row[2] != "")
            writer.writerows(rows)

    print(f"Processed {len(images)} image(s); recorded {detection_count} detection(s).")
    print(f"Annotated images and CSV: {args.output.resolve()}")
    print("For mAP, precision, recall, and confusion-matrix plots, run training/evaluate.py.")


if __name__ == "__main__":
    main()
