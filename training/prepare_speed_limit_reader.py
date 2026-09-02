"""Create a speed-limit number-recognition dataset from a Roboflow YOLO ZIP.

The detector classes remain unchanged.  Each annotated speed-limit bounding box
is cropped from its source image and kept in Roboflow's existing train, valid,
or test split so that the number reader can be evaluated independently.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "speed_limit_reader"
SPEED_CLASS_TO_VALUE = {
    "speed-limit-5": "5",
    "speed-limit-15": "15",
    "speed-limit-30": "30",
    "speed-limit-40": "40",
    "speed-limit-50": "50",
    "speed-limit-60": "60",
    "speed-limit-80": "80",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path, help="Roboflow YOLO export ZIP")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace a previously generated crop dataset.",
    )
    return parser.parse_args()


def read_config(archive: zipfile.ZipFile) -> dict:
    config_entries = [name for name in archive.namelist() if name.endswith("data.yaml")]
    if not config_entries:
        raise ValueError("The ZIP does not contain data.yaml.")
    config = yaml.safe_load(archive.read(config_entries[0]).decode("utf-8"))
    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[index] if index in names else names[str(index)] for index in range(len(names))]
    if len(names) != 63 or len(set(names)) != 63:
        raise ValueError(f"Expected 63 unique classes, found {len(names)}.")
    missing = sorted(set(SPEED_CLASS_TO_VALUE) - set(names))
    if missing:
        raise ValueError(f"Missing speed-limit classes: {missing}")
    config["names"] = names
    return config


def find_image_entry(entries: set[str], label_entry: str) -> str | None:
    label_path = PurePosixPath(label_entry)
    image_base = PurePosixPath(label_path.parts[0]) / "images" / label_path.stem
    for suffix in IMAGE_SUFFIXES:
        candidate = f"{image_base}{suffix}"
        if candidate in entries:
            return candidate
    return None


def square_crop(image: np.ndarray, box: tuple[float, float, float, float], padding: float) -> np.ndarray | None:
    height, width = image.shape[:2]
    center_x, center_y, box_width, box_height = box
    x1 = (center_x - box_width / 2) * width
    y1 = (center_y - box_height / 2) * height
    x2 = (center_x + box_width / 2) * width
    y2 = (center_y + box_height / 2) * height

    pad_x = max(3.0, (x2 - x1) * padding)
    pad_y = max(3.0, (y2 - y1) * padding)
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    if right - left < 8 or bottom - top < 8:
        return None

    crop = image[top:bottom, left:right]
    crop_height, crop_width = crop.shape[:2]
    side = max(crop_width, crop_height)
    canvas = np.full((side, side, 3), 238, dtype=np.uint8)
    offset_x = (side - crop_width) // 2
    offset_y = (side - crop_height) // 2
    canvas[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width] = crop
    return canvas


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path.resolve()
    output = args.output.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    if not 0 <= args.padding <= 0.5:
        raise ValueError("Padding must be between 0 and 0.5.")
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"Output already exists: {output}. Use --replace to rebuild it.")
        shutil.rmtree(output)

    crop_root = output / "crops"
    crop_root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    counts: Counter[tuple[str, str]] = Counter()
    skipped = Counter()

    with zipfile.ZipFile(zip_path) as archive:
        config = read_config(archive)
        names: list[str] = config["names"]
        entries = set(archive.namelist())
        label_entries = sorted(
            name
            for name in entries
            if PurePosixPath(name).suffix == ".txt"
            and len(PurePosixPath(name).parts) >= 3
            and PurePosixPath(name).parts[0] in {"train", "valid", "test"}
            and PurePosixPath(name).parts[1] == "labels"
        )

        for label_entry in label_entries:
            split = PurePosixPath(label_entry).parts[0]
            image_entry = find_image_entry(entries, label_entry)
            if image_entry is None:
                skipped["missing_image"] += 1
                continue
            label_lines = archive.read(label_entry).decode("utf-8").splitlines()
            speed_rows: list[tuple[int, str, tuple[float, float, float, float]]] = []
            for line_number, line in enumerate(label_lines, start=1):
                fields = line.strip().split()
                if len(fields) < 5:
                    continue
                class_id = int(float(fields[0]))
                if not 0 <= class_id < len(names):
                    raise ValueError(f"Invalid class ID in {label_entry}:{line_number}")
                class_name = names[class_id]
                if class_name not in SPEED_CLASS_TO_VALUE:
                    continue
                speed_rows.append((class_id, class_name, tuple(map(float, fields[1:5]))))
            if not speed_rows:
                continue

            encoded = np.frombuffer(archive.read(image_entry), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                skipped["unreadable_image"] += len(speed_rows)
                continue

            for box_index, (class_id, class_name, box) in enumerate(speed_rows):
                value = SPEED_CLASS_TO_VALUE[class_name]
                crop = square_crop(image, box, args.padding)
                if crop is None:
                    skipped["tiny_crop"] += 1
                    continue
                destination_dir = crop_root / split / value
                destination_dir.mkdir(parents=True, exist_ok=True)
                source_stem = PurePosixPath(image_entry).stem
                destination = destination_dir / f"{source_stem}__box{box_index}.jpg"
                if not cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                    raise OSError(f"Could not write {destination}")
                counts[(split, value)] += 1
                records.append(
                    {
                        "split": split,
                        "value": value,
                        "class_id": class_id,
                        "class_name": class_name,
                        "source_image": image_entry,
                        "source_label": label_entry,
                        "crop_path": destination.relative_to(output).as_posix(),
                        "source_width": image.shape[1],
                        "source_height": image.shape[0],
                        "crop_width": crop.shape[1],
                        "crop_height": crop.shape[0],
                    }
                )

    fieldnames = list(records[0]) if records else []
    if not records:
        raise ValueError("No speed-limit annotations were found.")
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "source_zip": zip_path.name,
        "source_zip_bytes": zip_path.stat().st_size,
        "classes": list(SPEED_CLASS_TO_VALUE.values()),
        "padding": args.padding,
        "total_crops": len(records),
        "counts": {
            split: {value: counts[(split, value)] for value in SPEED_CLASS_TO_VALUE.values()}
            for split in ("train", "valid", "test")
        },
        "skipped": dict(skipped),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
