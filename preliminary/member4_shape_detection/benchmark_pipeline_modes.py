#!/usr/bin/env python3
"""Benchmark three traffic-sign detection pipeline modes on identical input.

The benchmark compares:

* ``full_frame``: one YOLO inference on every complete frame.
* ``roi_only``: HSV/contour proposals followed by YOLO on proposal crops only.
* ``safe_hybrid``: ROI inference every frame plus periodic full-frame YOLO.

Input media is never modified. Results are written as JSON and CSV under the
directory supplied with ``--output``.

Native YOLO26 checkpoints keep their end-to-end, NMS-free prediction path.
``--legacy-nms-iou`` is applied only to older/non-end-to-end checkpoints;
pipeline ROI/full-frame merging and ground-truth matching use separate IoUs.

Ground-truth labels are optional. When ``--labels`` is supplied, labels must be
standard YOLO detection text files (``class x_center y_center width height``,
normalized to 0..1). For an image directory, the label tree mirrors the image
tree. For video, files are named ``frame_000000.txt``, ``frame_000001.txt``, and
so on (zero-based). Missing label files are treated as valid empty/no-object
frames, following the common YOLO convention.

Examples:

    python benchmark_pipeline_modes.py \
        --model models/best.pt --source test_images --labels test_labels \
        --output benchmark_results

    python benchmark_pipeline_modes.py \
        --model models/best.pt --source signs.mp4 \
        --no-sign-source backgrounds.mp4 \
        --output benchmark_results --imgsz 640 --roi-imgsz 640 --full-interval 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MODES = ("full_frame", "roi_only", "safe_hybrid")
METRIC_IOU = 0.5


@dataclass(frozen=True)
class FrameItem:
    """A decoded frame and the label path relative to a labels directory."""

    index: int
    name: str
    label_relative_path: Path
    image: np.ndarray


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]
    source: str


@dataclass
class MatchTotals:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ground_truth_boxes: int = 0


class FrameSource:
    """Re-openable, deterministic image-directory, image, or video source."""

    def __init__(
        self,
        source: Path,
        sequence_fps: float,
        max_frames: Optional[int],
    ) -> None:
        self.path = source.resolve()
        self.sequence_fps = sequence_fps
        self.max_frames = max_frames

        if not self.path.exists():
            raise FileNotFoundError(f"Input source does not exist: {self.path}")

        self.image_paths: List[Path] = []
        if self.path.is_dir():
            self.kind = "image_directory"
            self.image_paths = sorted(
                p
                for p in self.path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not self.image_paths:
                raise ValueError(f"No supported images found in: {self.path}")
            self.nominal_fps = sequence_fps
            self.duration_basis = "--sequence-fps assumption for image sequence"
        elif self.path.suffix.lower() in IMAGE_EXTENSIONS:
            self.kind = "single_image"
            self.image_paths = [self.path]
            self.nominal_fps = sequence_fps
            self.duration_basis = "--sequence-fps assumption for still image"
        else:
            self.kind = "video"
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                raise ValueError(f"OpenCV could not open video: {self.path}")
            detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            if math.isfinite(detected_fps) and detected_fps > 0:
                self.nominal_fps = detected_fps
                self.duration_basis = "video FPS metadata"
            else:
                self.nominal_fps = sequence_fps
                self.duration_basis = (
                    "--sequence-fps fallback because video FPS metadata was invalid"
                )

    def __iter__(self) -> Iterator[FrameItem]:
        if self.kind in {"image_directory", "single_image"}:
            yield from self._iter_images()
        else:
            yield from self._iter_video()

    def _iter_images(self) -> Iterator[FrameItem]:
        limit = self.max_frames if self.max_frames is not None else len(self.image_paths)
        for index, image_path in enumerate(self.image_paths[:limit]):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV could not decode image: {image_path}")
            if self.kind == "image_directory":
                relative = image_path.relative_to(self.path)
            else:
                relative = Path(image_path.name)
            yield FrameItem(
                index=index,
                name=str(relative),
                label_relative_path=relative.with_suffix(".txt"),
                image=image,
            )

    def _iter_video(self) -> Iterator[FrameItem]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not re-open video: {self.path}")
        index = 0
        try:
            while self.max_frames is None or index < self.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                label_name = f"frame_{index:06d}.txt"
                yield FrameItem(
                    index=index,
                    name=f"frame_{index:06d}",
                    label_relative_path=Path(label_name),
                    image=frame,
                )
                index += 1
        finally:
            capture.release()

    def first_frame(self) -> np.ndarray:
        try:
            return next(iter(self)).image
        except StopIteration as exc:
            raise ValueError(f"No decodable frames found in: {self.path}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-frame YOLO, HSV ROI-only, and periodic-full safe hybrid "
            "modes. JSON and CSV summaries are written without changing inputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Local YOLO checkpoint file or exported-model directory",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Primary video, image, or image directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for benchmark_summary.json and benchmark_summary.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional YOLO ground-truth label directory for the primary source",
    )
    parser.add_argument(
        "--no-sign-source",
        type=Path,
        help=(
            "Optional source known to contain no traffic signs; every detection is "
            "counted as noise"
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Full-frame YOLO size")
    parser.add_argument(
        "--roi-imgsz",
        type=int,
        default=640,
        help="YOLO size for candidate crops; fixed-size final exports require 640",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35, help="Full-frame confidence threshold"
    )
    parser.add_argument(
        "--roi-conf", type=float, default=0.30, help="ROI inference confidence threshold"
    )
    parser.add_argument(
        "--legacy-nms-iou",
        "--nms-iou",
        dest="legacy_nms_iou",
        type=float,
        default=0.45,
        help=(
            "per-image NMS IoU for older/non-end-to-end checkpoints; retained "
            "as --nms-iou alias and not passed to NMS-free YOLO26"
        ),
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.50,
        help="Class-aware IoU used to merge duplicate crop/full predictions",
    )
    parser.add_argument(
        "--roi-min-area",
        type=float,
        default=1200.0,
        help="Minimum HSV contour area in pixels",
    )
    parser.add_argument(
        "--roi-min-area-ratio",
        type=float,
        default=0.0,
        help="Minimum HSV contour area as a fraction of frame area",
    )
    parser.add_argument(
        "--roi-max-area-ratio",
        type=float,
        default=0.40,
        help="Maximum padded ROI area as a fraction of frame area",
    )
    parser.add_argument(
        "--roi-aspect-min", type=float, default=0.25, help="Minimum contour-box width/height"
    )
    parser.add_argument(
        "--roi-aspect-max", type=float, default=4.0, help="Maximum contour-box width/height"
    )
    parser.add_argument(
        "--roi-padding", type=int, default=24, help="Padding around each contour in pixels"
    )
    parser.add_argument(
        "--roi-min-side", type=int, default=24, help="Minimum padded ROI side in pixels"
    )
    parser.add_argument(
        "--roi-dedup-iou",
        type=float,
        default=0.65,
        help="IoU used to discard duplicate candidate regions",
    )
    parser.add_argument(
        "--roi-max-crops", type=int, default=2, help="Maximum ROI crops per frame"
    )
    parser.add_argument(
        "--full-interval",
        type=int,
        default=5,
        help="Safe-hybrid full-frame inference interval; frame zero is always included",
    )
    parser.add_argument(
        "--max-det", type=int, default=30, help="Maximum YOLO predictions per image"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, for example cpu, 0, or 0,1; auto-selected when omitted",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional primary-source frame limit",
    )
    parser.add_argument(
        "--no-sign-max-frames",
        type=int,
        default=None,
        help="Optional no-sign-source frame limit",
    )
    parser.add_argument(
        "--sequence-fps",
        type=float,
        default=30.0,
        help="Timing assumption for image sequences and fallback for invalid video metadata",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(
            f"Local model file/export does not exist: {args.model.resolve()}"
        )
    if args.labels is not None and not args.labels.is_dir():
        raise NotADirectoryError(f"Labels directory does not exist: {args.labels.resolve()}")
    if args.imgsz <= 0 or args.roi_imgsz <= 0:
        raise ValueError("--imgsz and --roi-imgsz must be positive")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive")
    if args.full_interval <= 0:
        raise ValueError("--full-interval must be positive")
    if args.roi_max_crops <= 0:
        raise ValueError("--roi-max-crops must be positive")
    if args.roi_padding < 0:
        raise ValueError("--roi-padding cannot be negative")
    if args.roi_min_side <= 0:
        raise ValueError("--roi-min-side must be positive")
    if args.roi_min_area < 0:
        raise ValueError("--roi-min-area cannot be negative")
    if args.roi_aspect_min <= 0 or args.roi_aspect_max < args.roi_aspect_min:
        raise ValueError("ROI aspect thresholds are invalid")
    if args.sequence_fps <= 0:
        raise ValueError("--sequence-fps must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when supplied")
    if args.no_sign_max_frames is not None and args.no_sign_max_frames <= 0:
        raise ValueError("--no-sign-max-frames must be positive when supplied")
    for name in (
        "conf",
        "roi_conf",
        "legacy_nms_iou",
        "merge_iou",
        "roi_dedup_iou",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if not 0.0 <= args.roi_min_area_ratio <= 1.0:
        raise ValueError("--roi-min-area-ratio must be between 0 and 1")
    if not 0.0 < args.roi_max_area_ratio <= 1.0:
        raise ValueError("--roi-max-area-ratio must be greater than 0 and at most 1")
    if args.roi_max_area_ratio < args.roi_min_area_ratio:
        raise ValueError("--roi-max-area-ratio cannot be below --roi-min-area-ratio")


def compute_iou(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def model_uses_nms_free_prediction(model: YOLO) -> bool:
    """Return whether the native checkpoint or initialized export is end-to-end."""
    core_model = getattr(model, "model", None)
    if bool(getattr(core_model, "end2end", False)):
        return True
    predictor = getattr(model, "predictor", None)
    backend_model = getattr(predictor, "model", None)
    return bool(getattr(backend_model, "end2end", False))


def add_model_postprocessing_options(
    options: Dict[str, object], model: YOLO, args: argparse.Namespace
) -> Dict[str, object]:
    """Apply legacy per-image NMS only when the checkpoint actually uses it."""
    if not model_uses_nms_free_prediction(model):
        options["iou"] = args.legacy_nms_iou
    return options


def find_colored_regions(frame: np.ndarray, args: argparse.Namespace) -> List[Tuple[int, int, int, int]]:
    """Return padded x1/y1/x2/y2 candidate regions in descending contour area."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([45, 255, 255]))
    mask = red_low | red_high | blue | yellow

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_height, frame_width = frame.shape[:2]
    frame_area = float(frame_height * frame_width)
    minimum_area = max(args.roi_min_area, args.roi_min_area_ratio * frame_area)
    maximum_roi_area = args.roi_max_area_ratio * frame_area
    regions: List[Tuple[int, int, int, int]] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        contour_area = float(cv2.contourArea(contour))
        if contour_area < minimum_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        raw_aspect = width / max(height, 1)
        if not args.roi_aspect_min <= raw_aspect <= args.roi_aspect_max:
            continue
        x1 = max(0, x - args.roi_padding)
        y1 = max(0, y - args.roi_padding)
        x2 = min(frame_width, x + width + args.roi_padding)
        y2 = min(frame_height, y + height + args.roi_padding)
        padded_width = x2 - x1
        padded_height = y2 - y1
        if padded_width < args.roi_min_side or padded_height < args.roi_min_side:
            continue
        if padded_width * padded_height > maximum_roi_area:
            continue
        candidate = (x1, y1, x2, y2)
        if any(
            compute_iou(candidate, existing) >= args.roi_dedup_iou
            for existing in regions
        ):
            continue
        regions.append(candidate)
        if len(regions) >= args.roi_max_crops:
            break
    return regions


def predict(
    model: YOLO,
    image: np.ndarray,
    confidence: float,
    source_name: str,
    args: argparse.Namespace,
    image_size: Optional[int] = None,
) -> List[Detection]:
    options = {
        "source": image,
        "imgsz": image_size if image_size is not None else args.imgsz,
        "conf": confidence,
        "max_det": args.max_det,
        "verbose": False,
    }
    add_model_postprocessing_options(options, model, args)
    if args.device is not None:
        options["device"] = args.device
    result = model.predict(**options)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    return [
        Detection(
            class_id=int(class_id),
            confidence=float(score),
            bbox=tuple(float(value) for value in box),
            source=source_name,
        )
        for box, class_id, score in zip(xyxy, classes, confidences)
    ]


def predict_regions(
    model: YOLO,
    frame: np.ndarray,
    regions: Sequence[Tuple[int, int, int, int]],
    args: argparse.Namespace,
) -> Tuple[List[Detection], int]:
    crops: List[np.ndarray] = []
    offsets: List[Tuple[int, int]] = []
    for x1, y1, x2, y2 in regions:
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        crops.append(crop)
        offsets.append((x1, y1))

    if not crops:
        return [], 0

    options = {
        "imgsz": args.roi_imgsz,
        "conf": args.roi_conf,
        "max_det": args.max_det,
        "verbose": False,
    }
    add_model_postprocessing_options(options, model, args)
    if args.device is not None:
        options["device"] = args.device

    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    dynamic_or_pytorch = bool(getattr(backend, "dynamic", False)) or bool(
        getattr(backend, "pt", False)
    )
    backend_batch = int(getattr(backend, "batch", 1) or 1)
    if dynamic_or_pytorch or backend_batch >= len(crops):
        results = model.predict(source=crops, **options)
    else:
        # Fixed-batch ONNX/OpenVINO exports are evaluated one crop at a time.
        results = []
        for crop in crops:
            results.extend(model.predict(source=crop, **options))

    detections: List[Detection] = []
    for result, (x1, y1) in zip(results, offsets):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        for box, class_id, score in zip(xyxy, classes, confidences):
            bx1, by1, bx2, by2 = (float(value) for value in box)
            detections.append(
                Detection(
                    class_id=int(class_id),
                    confidence=float(score),
                    bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                    source="roi",
                )
            )
    return detections, len(crops)


def merge_detections(
    detections: Sequence[Detection], merge_iou: float
) -> List[Detection]:
    """Confidence-first, class-aware suppression across ROI/full predictions."""

    kept: List[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            candidate.class_id == existing.class_id
            and compute_iou(candidate.bbox, existing.bbox) >= merge_iou
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def read_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Detection]:
    if not label_path.exists():
        return []
    labels: List[Detection] = []
    with label_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"Expected 5 YOLO detection fields in {label_path}:{line_number}; "
                    f"found {len(fields)}"
                )
            try:
                class_value, center_x, center_y, width, height = map(float, fields)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric YOLO label in {label_path}:{line_number}"
                ) from exc
            if not class_value.is_integer() or class_value < 0:
                raise ValueError(f"Invalid class ID in {label_path}:{line_number}")
            normalized = (center_x, center_y, width, height)
            if not all(math.isfinite(value) for value in normalized):
                raise ValueError(f"Non-finite YOLO box in {label_path}:{line_number}")
            if not all(0.0 <= value <= 1.0 for value in normalized):
                raise ValueError(
                    f"YOLO coordinates must be normalized to 0..1 in "
                    f"{label_path}:{line_number}"
                )
            if width == 0 or height == 0:
                raise ValueError(f"Zero-size YOLO box in {label_path}:{line_number}")
            x1 = max(0.0, (center_x - width / 2.0) * image_width)
            y1 = max(0.0, (center_y - height / 2.0) * image_height)
            x2 = min(float(image_width), (center_x + width / 2.0) * image_width)
            y2 = min(float(image_height), (center_y + height / 2.0) * image_height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"YOLO box is outside the image in {label_path}:{line_number}")
            labels.append(
                Detection(
                    class_id=int(class_value),
                    confidence=1.0,
                    bbox=(x1, y1, x2, y2),
                    source="ground_truth",
                )
            )
    return labels


def update_matches(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    totals: MatchTotals,
) -> None:
    unmatched = set(range(len(ground_truth)))
    totals.ground_truth_boxes += len(ground_truth)

    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        best_index: Optional[int] = None
        best_iou = -1.0
        for index in unmatched:
            target = ground_truth[index]
            if target.class_id != prediction.class_id:
                continue
            overlap = compute_iou(prediction.bbox, target.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= METRIC_IOU:
            totals.true_positives += 1
            unmatched.remove(best_index)
        else:
            totals.false_positives += 1
    totals.false_negatives += len(unmatched)


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator > 0 else None


def ground_truth_metrics(totals: MatchTotals) -> Dict[str, object]:
    precision = safe_ratio(
        totals.true_positives,
        totals.true_positives + totals.false_positives,
    )
    recall = safe_ratio(
        totals.true_positives,
        totals.true_positives + totals.false_negatives,
    )
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "metric": "class-aware object detection at IoU 0.5",
        "iou_threshold": METRIC_IOU,
        **asdict(totals),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_mode(
    mode: str,
    source: FrameSource,
    model: YOLO,
    args: argparse.Namespace,
    labels_root: Optional[Path],
    dataset_role: str,
) -> Dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")

    processed_frames = 0
    total_detections = 0
    frames_with_detections = 0
    roi_candidates = 0
    yolo_calls = 0
    full_frame_calls = 0
    crop_calls = 0
    processing_seconds = 0.0
    labels_found = 0
    labels_missing = 0
    match_totals = MatchTotals()

    wall_start = time.perf_counter()
    for item in source:
        processing_start = time.perf_counter()
        detections: List[Detection]

        if mode == "full_frame":
            detections = predict(model, item.image, args.conf, "full_frame", args)
            yolo_calls += 1
            full_frame_calls += 1
        else:
            regions = find_colored_regions(item.image, args)
            roi_candidates += len(regions)
            roi_detections, calls = predict_regions(model, item.image, regions, args)
            yolo_calls += calls
            crop_calls += calls
            combined = list(roi_detections)

            if mode == "safe_hybrid" and item.index % args.full_interval == 0:
                combined.extend(
                    predict(model, item.image, args.conf, "periodic_full_frame", args)
                )
                yolo_calls += 1
                full_frame_calls += 1

            # Deliberately no forced full-frame fallback when the ROI list is empty.
            detections = merge_detections(combined, args.merge_iou)

        processing_seconds += time.perf_counter() - processing_start
        processed_frames += 1
        total_detections += len(detections)
        if detections:
            frames_with_detections += 1

        if labels_root is not None:
            label_path = labels_root / item.label_relative_path
            if label_path.is_file():
                labels_found += 1
            else:
                labels_missing += 1
            height, width = item.image.shape[:2]
            targets = read_yolo_labels(label_path, width, height)
            update_matches(detections, targets, match_totals)

    elapsed_seconds = time.perf_counter() - wall_start
    if processed_frames == 0:
        raise ValueError(f"No frames were processed from: {source.path}")

    input_duration_seconds = processed_frames / source.nominal_fps
    result: Dict[str, object] = {
        "dataset_role": dataset_role,
        "mode": mode,
        "model_prediction_postprocessing": (
            "native_end_to_end_nms_free"
            if model_uses_nms_free_prediction(model)
            else "traditional_per_image_nms"
        ),
        "model_nms_iou_applied": (
            None
            if model_uses_nms_free_prediction(model)
            else args.legacy_nms_iou
        ),
        "source": str(source.path),
        "source_kind": source.kind,
        "nominal_input_fps": source.nominal_fps,
        "duration_basis": source.duration_basis,
        "input_duration_seconds": input_duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "processing_seconds": processing_seconds,
        "processed_frames": processed_frames,
        "end_to_end_fps": processed_frames / elapsed_seconds if elapsed_seconds > 0 else None,
        "processing_fps": (
            processed_frames / processing_seconds if processing_seconds > 0 else None
        ),
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "frame_detection_coverage": frames_with_detections / processed_frames,
        "roi_candidates": roi_candidates,
        "yolo_calls": yolo_calls,
        "full_frame_calls": full_frame_calls,
        "crop_calls": crop_calls,
    }

    if labels_root is not None:
        result["evaluation"] = {
            "kind": "ground_truth_metrics",
            "labels_directory": str(labels_root),
            "label_files_found": labels_found,
            "missing_label_files_treated_as_empty": labels_missing,
            **ground_truth_metrics(match_totals),
        }
    else:
        result["evaluation"] = {
            "kind": "performance_and_detection_coverage_only",
            "message": (
                "No ground-truth labels were supplied; precision, recall, and F1 "
                "were not computed."
            ),
        }

    if dataset_role == "known_no_sign":
        duration_minutes = input_duration_seconds / 60.0
        result["noise"] = {
            "assumption": "--no-sign-source is known to contain zero traffic signs",
            "false_detections": total_detections,
            "false_detections_per_input_minute": safe_ratio(
                total_detections, duration_minutes
            ),
            "false_positive_frames": frames_with_detections,
            "false_positive_frames_per_input_minute": safe_ratio(
                frames_with_detections, duration_minutes
            ),
        }
    return result


def flatten_result(result: Dict[str, object]) -> Dict[str, object]:
    evaluation = result.get("evaluation", {})
    noise = result.get("noise", {})
    return {
        "dataset_role": result.get("dataset_role"),
        "mode": result.get("mode"),
        "model_prediction_postprocessing": result.get(
            "model_prediction_postprocessing"
        ),
        "model_nms_iou_applied": result.get("model_nms_iou_applied"),
        "source": result.get("source"),
        "source_kind": result.get("source_kind"),
        "nominal_input_fps": result.get("nominal_input_fps"),
        "duration_basis": result.get("duration_basis"),
        "input_duration_seconds": result.get("input_duration_seconds"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "processing_seconds": result.get("processing_seconds"),
        "processed_frames": result.get("processed_frames"),
        "end_to_end_fps": result.get("end_to_end_fps"),
        "processing_fps": result.get("processing_fps"),
        "total_detections": result.get("total_detections"),
        "frames_with_detections": result.get("frames_with_detections"),
        "frame_detection_coverage": result.get("frame_detection_coverage"),
        "roi_candidates": result.get("roi_candidates"),
        "yolo_calls": result.get("yolo_calls"),
        "full_frame_calls": result.get("full_frame_calls"),
        "crop_calls": result.get("crop_calls"),
        "evaluation_kind": evaluation.get("kind"),
        "label_files_found": evaluation.get("label_files_found"),
        "missing_label_files_treated_as_empty": evaluation.get(
            "missing_label_files_treated_as_empty"
        ),
        "metric_iou": evaluation.get("iou_threshold"),
        "ground_truth_boxes": evaluation.get("ground_truth_boxes"),
        "true_positives": evaluation.get("true_positives"),
        "false_positives": evaluation.get("false_positives"),
        "false_negatives": evaluation.get("false_negatives"),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
        "false_detections": noise.get("false_detections"),
        "false_detections_per_input_minute": noise.get(
            "false_detections_per_input_minute"
        ),
        "false_positive_frames": noise.get("false_positive_frames"),
        "false_positive_frames_per_input_minute": noise.get(
            "false_positive_frames_per_input_minute"
        ),
    }


def write_outputs(
    output_directory: Path,
    report: Dict[str, object],
    results: Sequence[Dict[str, object]],
) -> Tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "benchmark_summary.json"
    csv_path = output_directory / "benchmark_summary.csv"

    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    rows = [flatten_result(result) for result in results]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def print_result(result: Dict[str, object]) -> None:
    fps = result["end_to_end_fps"]
    coverage = result["frame_detection_coverage"]
    message = (
        f"  {result['mode']:<12} frames={result['processed_frames']:<6} "
        f"FPS={fps:.2f} detections={result['total_detections']:<6} "
        f"frame_coverage={coverage:.3f}"
    )
    evaluation = result["evaluation"]
    if evaluation["kind"] == "ground_truth_metrics":
        precision = evaluation["precision"]
        recall = evaluation["recall"]
        f1 = evaluation["f1"]
        formatted = lambda value: "n/a" if value is None else f"{value:.3f}"
        message += (
            f" precision={formatted(precision)} recall={formatted(recall)} "
            f"F1={formatted(f1)}"
        )
    if "noise" in result:
        rate = result["noise"]["false_detections_per_input_minute"]
        message += f" false_detections/min={rate:.3f}" if rate is not None else ""
    print(message)


def main() -> None:
    args = parse_args()
    validate_args(args)

    primary_source = FrameSource(args.source, args.sequence_fps, args.max_frames)
    labels_root = args.labels.resolve() if args.labels is not None else None
    no_sign_source = (
        FrameSource(args.no_sign_source, args.sequence_fps, args.no_sign_max_frames)
        if args.no_sign_source is not None
        else None
    )

    print(f"Loading model: {args.model.resolve()}")
    model = YOLO(str(args.model.resolve()), task="detect")
    print("Warming up one excluded inference...")
    predict(model, primary_source.first_frame(), args.conf, "warmup", args)
    nms_free_prediction = model_uses_nms_free_prediction(model)
    if nms_free_prediction:
        print("Prediction mode: native end-to-end, NMS-free (YOLO26-ready).")
    else:
        print(
            "Prediction mode: traditional per-image NMS "
            f"(IoU {args.legacy_nms_iou:.2f})."
        )

    all_results: List[Dict[str, object]] = []
    print(f"Primary source: {primary_source.path}")
    if labels_root is None:
        print("  Reporting performance and detection coverage only (no labels supplied).")
    for mode in MODES:
        result = run_mode(
            mode,
            primary_source,
            model,
            args,
            labels_root,
            dataset_role="primary",
        )
        all_results.append(result)
        print_result(result)

    if no_sign_source is not None:
        print(f"Known no-sign source: {no_sign_source.path}")
        for mode in MODES:
            result = run_mode(
                mode,
                no_sign_source,
                model,
                args,
                labels_root=None,
                dataset_role="known_no_sign",
            )
            all_results.append(result)
            print_result(result)

    report: Dict[str, object] = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "configuration": {
            "full_frame_imgsz": args.imgsz,
            "roi_imgsz": args.roi_imgsz,
            "full_frame_confidence": args.conf,
            "roi_confidence": args.roi_conf,
            "model_prediction_postprocessing": (
                "native_end_to_end_nms_free"
                if nms_free_prediction
                else "traditional_per_image_nms"
            ),
            "model_end_to_end_nms_free": nms_free_prediction,
            "legacy_model_nms_iou_configured": args.legacy_nms_iou,
            "model_nms_iou_applied": (
                None if nms_free_prediction else args.legacy_nms_iou
            ),
            "cross_source_merge_iou": args.merge_iou,
            "metric_iou": METRIC_IOU,
            "roi_min_area_pixels": args.roi_min_area,
            "roi_min_area_ratio": args.roi_min_area_ratio,
            "roi_max_area_ratio": args.roi_max_area_ratio,
            "roi_aspect_min": args.roi_aspect_min,
            "roi_aspect_max": args.roi_aspect_max,
            "roi_padding_pixels": args.roi_padding,
            "roi_min_side_pixels": args.roi_min_side,
            "roi_dedup_iou": args.roi_dedup_iou,
            "roi_max_crops": args.roi_max_crops,
            "safe_hybrid_full_interval": args.full_interval,
            "max_detections_per_inference": args.max_det,
            "device": args.device if args.device is not None else "Ultralytics auto",
            "primary_max_frames": args.max_frames,
            "no_sign_max_frames": args.no_sign_max_frames,
            "sequence_fps": args.sequence_fps,
        },
        "interpretation": {
            "without_labels": (
                "Only throughput, detections, and frame detection coverage are reported."
            ),
            "with_labels": (
                "Precision, recall, and F1 use greedy class-aware matching at IoU 0.5."
            ),
            "noise_rate": (
                "False detections per input minute uses media duration, not benchmark "
                "wall-clock duration. For image sequences, duration uses --sequence-fps."
            ),
            "roi_behavior": (
                "ROI-only never falls back to full-frame inference. Safe hybrid uses "
                "full-frame inference only at the configured periodic interval."
            ),
            "temporal_scope": (
                "This benchmark compares the raw detector stage before the webcam "
                "demo's multi-frame confirmation, so temporal filtering cannot hide "
                "detector misses or one-frame false positives."
            ),
            "iou_scope": (
                "YOLO26 uses its native NMS-free output, so the legacy model NMS IoU "
                "is not passed. Cross-source merge IoU and metric IoU remain active "
                "because they serve separate pipeline and evaluation purposes."
                if nms_free_prediction
                else "The legacy model NMS IoU is applied inside each inference. "
                "Cross-source merge IoU and metric IoU are separate settings."
            ),
        },
        "results": all_results,
    }
    json_path, csv_path = write_outputs(args.output.resolve(), report, all_results)
    print(f"JSON summary: {json_path}")
    print(f"CSV summary:  {csv_path}")


if __name__ == "__main__":
    main()
