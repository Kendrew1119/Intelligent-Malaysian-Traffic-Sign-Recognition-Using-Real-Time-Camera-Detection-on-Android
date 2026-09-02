# ============================================
# [Member 4] webcam_yolo_demo.py - SAFE HYBRID VERSION
# ============================================
# Purpose: Compare two laptop-camera inference modes:
#
#   FULL:
#     YOLO runs on the complete frame every frame.
#
#   HYBRID:
#     OpenCV finds color-based candidate regions in the current frame.
#     YOLO runs on the raw candidate crops (batched), while a periodic
#     full-frame YOLO scan recovers signs that HSV preprocessing misses.
#
# OpenCV regions are proposals only. They never become detections unless
# YOLO detects a class in the crop. A small IoU tracker confirms a result
# only after the same class appears repeatedly at a similar location.
#
# Controls:
#   q       quit
#   s       save a screenshot
#   m       toggle HSV mask/ROI debug overlay
#   h       toggle FULL/HYBRID mode
#   u       toggle unconfirmed YOLO candidates
#   [ / ]   lower/raise both YOLO confidence thresholds
#
# Usage:
#   pip install ultralytics opencv-python numpy
#   python webcam_yolo_demo.py
# ============================================

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================
# CONFIGURATION - CALIBRATE ON HELD-OUT VIDEO
# ============================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "best.pt"
DEFAULT_CAMERA_INDEX = 0
EXPECTED_CLASS_COUNT = 63
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
SCREENSHOT_DIR = "screenshots"

# YOLO settings. The final ONNX/OpenVINO exports have a fixed 640px input, so
# full frames and candidate crops must use the same exported input size.
# Ultralytics still performs aspect-preserving letterbox preprocessing.
FULL_IMGSZ = 640
CROP_IMGSZ = 640
FULL_CONF_THRESHOLD = 0.35
CROP_CONF_THRESHOLD = 0.30
CONFIDENCE_STEP = 0.05
# YOLO26 uses its native end-to-end, NMS-free prediction head. This threshold
# is passed only to older/non-end-to-end Ultralytics detection checkpoints.
# It is unrelated to ROI/full-frame merging and temporal track matching below.
LEGACY_MODEL_NMS_IOU = 0.45
MAX_DETECTIONS_PER_IMAGE = 30

# Hybrid scheduling. A crop plus a full scan every fifth frame can improve
# small-sign coverage, but the fixed-size export does not guarantee a speedup.
# Actual performance must be measured on the deployment laptop.
HYBRID_FULL_EVERY_N_FRAMES = 5
HYBRID_ROI_EVERY_N_FRAMES = 1
MAX_CROPS_PER_FRAME = 2

# OpenCV proposal filtering. These settings affect recall/speed only;
# they do not create class predictions.
MIN_CONTOUR_AREA = 1200
MIN_CROP_SIDE = 24
MAX_ROI_AREA_RATIO = 0.40
MAX_ROI_ASPECT_RATIO = 4.0
CROP_PADDING = 24
ROI_DEDUP_IOU = 0.65

# Merge and temporal confirmation.
MERGE_IOU_THRESHOLD = 0.50
TRACK_IOU_THRESHOLD = 0.30
CONFIRMATION_HITS = 3
# This permits consecutive periodic full scans to confirm a sign in hybrid
# mode, while still expiring detections that disappear.
CONFIRM_MAX_GAP_FRAMES = HYBRID_FULL_EVERY_N_FRAMES + 1
TRACK_MAX_AGE_FRAMES = (2 * HYBRID_FULL_EVERY_N_FRAMES) + 1

# Production-like default: do not draw one-frame YOLO candidates. Press "u"
# while calibrating to inspect them.
SHOW_UNCONFIRMED_DEFAULT = False

# HSV color ranges used only to propose crop regions.
RED_LOWER_1 = np.array([0, 70, 50])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 70, 50])
RED_UPPER_2 = np.array([180, 255, 255])
BLUE_LOWER = np.array([90, 50, 50])
BLUE_UPPER = np.array([130, 255, 255])
YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([45, 255, 255])


def compute_iou(box1, box2):
    """Return IoU for two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / max(union, 1e-6)


def detect_colored_regions(img):
    """Return current-frame HSV proposals and the combined debug mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    frame_h, frame_w = img.shape[:2]
    frame_area = float(frame_h * frame_w)

    mask_red = (
        cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        | cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    )
    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    combined = mask_red | mask_blue | mask_yellow

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, kernel, iterations=1
    )

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        raw_x, raw_y, raw_w, raw_h = cv2.boundingRect(contour)
        raw_aspect = raw_w / max(raw_h, 1)
        if (
            raw_aspect > MAX_ROI_ASPECT_RATIO
            or raw_aspect < 1.0 / MAX_ROI_ASPECT_RATIO
        ):
            continue

        x1 = max(0, raw_x - CROP_PADDING)
        y1 = max(0, raw_y - CROP_PADDING)
        x2 = min(frame_w, raw_x + raw_w + CROP_PADDING)
        y2 = min(frame_h, raw_y + raw_h + CROP_PADDING)
        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < MIN_CROP_SIDE or crop_h < MIN_CROP_SIDE:
            continue
        if (crop_w * crop_h) / frame_area > MAX_ROI_AREA_RATIO:
            continue

        candidate_box = (x1, y1, x2, y2)
        if any(
            compute_iou(candidate_box, existing) >= ROI_DEDUP_IOU
            for existing in regions
        ):
            continue

        regions.append(candidate_box)
        if len(regions) >= MAX_CROPS_PER_FRAME:
            break

    return regions, combined


def class_name(model, class_id):
    """Read a class name from either list-style or dict-style YOLO names."""
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def model_uses_nms_free_prediction(model):
    """Return whether the native checkpoint or initialized export is end-to-end."""
    core_model = getattr(model, "model", None)
    if bool(getattr(core_model, "end2end", False)):
        return True
    predictor = getattr(model, "predictor", None)
    backend_model = getattr(predictor, "model", None)
    return bool(getattr(backend_model, "end2end", False))


def prediction_options(model, source, image_size, confidence):
    """Build inference options without applying legacy NMS to YOLO26."""
    options = {
        "source": source,
        "imgsz": image_size,
        "conf": confidence,
        "max_det": MAX_DETECTIONS_PER_IMAGE,
        "verbose": False,
    }
    if not model_uses_nms_free_prediction(model):
        options["iou"] = LEGACY_MODEL_NMS_IOU
    return options


def detections_from_result(model, result, source, offset_x=0, offset_y=0):
    """Convert an Ultralytics result into the demo's detection dictionaries."""
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "cls_id": class_id,
                "cls_name": class_name(model, class_id),
                "conf": float(box.conf[0]),
                "bbox": (
                    int(round(x1)) + offset_x,
                    int(round(y1)) + offset_y,
                    int(round(x2)) + offset_x,
                    int(round(y2)) + offset_y,
                ),
                "source": source,
            }
        )
    return detections


def run_yolo_full_frame(model, frame, confidence):
    """Run one aspect-preserving YOLO inference on the complete frame."""
    results = model.predict(
        **prediction_options(
            model,
            source=frame,
            image_size=FULL_IMGSZ,
            confidence=confidence,
        )
    )
    return detections_from_result(model, results[0], "full")


def backend_accepts_batch(model, requested_batch):
    """Return whether the initialized backend accepts this crop batch size."""
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    if backend is None:
        return requested_batch == 1
    if bool(getattr(backend, "pt", False)) or bool(
        getattr(backend, "dynamic", False)
    ):
        return True
    return int(getattr(backend, "batch", 1) or 1) >= requested_batch


def run_yolo_on_regions(model, frame, regions, confidence):
    """Infer raw ROI crops and map original-crop coordinates to the frame."""
    crops = []
    valid_regions = []

    for x1, y1, x2, y2 in regions:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < MIN_CROP_SIDE:
            continue
        # No cv2.resize here. Ultralytics letterboxes without shape distortion.
        crops.append(crop)
        valid_regions.append((x1, y1, x2, y2))

    if not crops:
        return []

    if backend_accepts_batch(model, len(crops)):
        results = model.predict(
            **prediction_options(
                model,
                source=crops,
                image_size=CROP_IMGSZ,
                confidence=confidence,
            )
        )
    else:
        # The canonical ONNX/OpenVINO exports use a fixed batch size of one.
        results = []
        for crop in crops:
            results.extend(
                model.predict(
                    **prediction_options(
                        model,
                        source=crop,
                        image_size=CROP_IMGSZ,
                        confidence=confidence,
                    )
                )
            )

    detections = []
    for result, region in zip(results, valid_regions):
        x1, y1, _, _ = region
        detections.extend(
            detections_from_result(
                model, result, "crop", offset_x=x1, offset_y=y1
            )
        )
    return detections


def merge_detections(full_detections, crop_detections):
    """Merge current-frame full/crop results with pipeline IoU de-duplication."""
    candidates = sorted(
        full_detections + crop_detections,
        key=lambda detection: detection["conf"],
        reverse=True,
    )
    merged = []

    for candidate in candidates:
        duplicate_index = None
        for index, kept in enumerate(merged):
            if compute_iou(candidate["bbox"], kept["bbox"]) >= MERGE_IOU_THRESHOLD:
                duplicate_index = index
                break

        if duplicate_index is None:
            merged.append(candidate.copy())
            continue

        kept = merged[duplicate_index]
        if (
            candidate["cls_id"] == kept["cls_id"]
            and candidate["source"] != kept["source"]
        ):
            kept["source"] = "full+crop"

    return merged


class TemporalConfirmer:
    """Small class-aware IoU tracker used only for temporal confirmation."""

    def __init__(
        self,
        required_hits,
        match_iou,
        max_gap_frames,
        max_age_frames,
    ):
        self.required_hits = required_hits
        self.match_iou = match_iou
        self.max_gap_frames = max_gap_frames
        self.max_age_frames = max_age_frames
        self.tracks = {}
        self.next_track_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1

    def update(self, detections, frame_index):
        """Attach current detections to tracks; never return stale track boxes."""
        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if frame_index - track["last_seen_frame"] > self.max_age_frames
        ]
        for track_id in expired_ids:
            del self.tracks[track_id]

        used_track_ids = set()
        tracked_detections = []

        for detection in sorted(
            detections, key=lambda item: item["conf"], reverse=True
        ):
            best_track_id = None
            best_iou = self.match_iou

            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                if track["cls_id"] != detection["cls_id"]:
                    continue

                overlap = compute_iou(detection["bbox"], track["bbox"])
                if overlap >= best_iou:
                    best_iou = overlap
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                track = {
                    "cls_id": detection["cls_id"],
                    "bbox": detection["bbox"],
                    "hits": 1,
                    "last_seen_frame": frame_index,
                    "confirmed": self.required_hits <= 1,
                }
                self.tracks[track_id] = track
                newly_confirmed = track["confirmed"]
            else:
                track_id = best_track_id
                track = self.tracks[track_id]
                previous_confirmed = track["confirmed"]
                frame_gap = frame_index - track["last_seen_frame"]

                if frame_gap <= self.max_gap_frames:
                    track["hits"] += 1
                else:
                    # A long gap starts a new confirmation sequence.
                    track["hits"] = 1
                    track["confirmed"] = False

                track["bbox"] = detection["bbox"]
                track["last_seen_frame"] = frame_index
                track["confirmed"] = track["hits"] >= self.required_hits
                newly_confirmed = track["confirmed"] and not previous_confirmed

            used_track_ids.add(track_id)
            current = detection.copy()
            current["track_id"] = track_id
            current["confirmation_hits"] = min(
                track["hits"], self.required_hits
            )
            current["confirmed"] = track["confirmed"]
            current["newly_confirmed"] = newly_confirmed
            tracked_detections.append(current)

        return tracked_detections


def draw_detection(frame, detection, show_unconfirmed):
    """Draw a current-frame YOLO result if its confirmation state allows it."""
    if not detection["confirmed"] and not show_unconfirmed:
        return

    x1, y1, x2, y2 = detection["bbox"]
    x1 = max(0, min(frame.shape[1] - 1, x1))
    y1 = max(0, min(frame.shape[0] - 1, y1))
    x2 = max(0, min(frame.shape[1] - 1, x2))
    y2 = max(0, min(frame.shape[0] - 1, y2))

    if not detection["confirmed"]:
        color = (0, 165, 255)
        thickness = 1
        state_text = (
            f"candidate {detection['confirmation_hits']}/{CONFIRMATION_HITS}"
        )
    elif "crop" in detection["source"]:
        color = (255, 255, 0)
        thickness = 3
        state_text = "confirmed"
    else:
        color = (0, 255, 0)
        thickness = 2
        state_text = "confirmed"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = (
        f"{detection['cls_name']} {detection['conf']:.0%} "
        f"[{state_text}]"
    )
    label_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    label_top = max(0, y1 - label_size[1] - baseline - 6)
    label_right = min(frame.shape[1] - 1, x1 + label_size[0] + 6)
    cv2.rectangle(
        frame, (x1, label_top), (label_right, y1), color, -1
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, max(label_size[1] + 1, y1 - baseline - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full-frame or safe-hybrid YOLO on a laptop webcam."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="YOLO model or export path; defaults to the canonical tuned 63-class model",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="OpenCV camera index",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"ERROR: Model or export '{model_path}' not found.")
        sys.exit(1)

    print("=" * 68)
    print(" MYSignVoice - Safe Hybrid Laptop Webcam Demo")
    print("=" * 68)
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path), task="detect")
    print("Model loaded successfully.")

    # Exported backends expose their end-to-end metadata only after predictor
    # initialization. This excluded warmup also removes cold-start cost from the
    # live FPS display. A legacy IoU option on this first call is ignored by an
    # end-to-end export; all later calls omit it once the backend is identified.
    print("Warming up one excluded inference...")
    warmup_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    model.predict(
        **prediction_options(
            model, warmup_frame, FULL_IMGSZ, FULL_CONF_THRESHOLD
        )
    )
    if model_uses_nms_free_prediction(model):
        print("Prediction mode: native end-to-end, NMS-free (YOLO26-ready).")
    else:
        print(
            "Prediction mode: traditional per-image NMS "
            f"(IoU {LEGACY_MODEL_NMS_IOU:.2f})."
        )
    class_count = len(model.names)
    if class_count != EXPECTED_CLASS_COUNT:
        print(
            f"WARNING: This model has {class_count} classes; the final contract "
            f"requires {EXPECTED_CLASS_COUNT}. Use it only for pipeline testing."
        )

    print(f"Opening camera (index {args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    # These are requests; webcam drivers may choose the closest supported mode.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    print(
        f"Camera requested {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS; "
        f"driver reports {actual_width}x{actual_height} @ {actual_fps:.1f} FPS."
    )
    print(
        f"Confirmation: {CONFIRMATION_HITS} hits, "
        f"class match + IoU >= {TRACK_IOU_THRESHOLD:.2f}."
    )
    print("  q=quit | s=screenshot | m=mask | h=mode | u=candidates")
    print("  [=lower confidence | ]=raise confidence")
    print("-" * 68)

    frame_count = 0
    screenshot_count = 0
    hybrid_mode = True
    show_mask = False
    show_unconfirmed = SHOW_UNCONFIRMED_DEFAULT
    full_confidence = FULL_CONF_THRESHOLD
    crop_confidence = CROP_CONF_THRESHOLD
    detection_history = []
    fps_ema = 0.0
    start_time = time.perf_counter()

    confirmer = TemporalConfirmer(
        required_hits=CONFIRMATION_HITS,
        match_iou=TRACK_IOU_THRESHOLD,
        max_gap_frames=CONFIRM_MAX_GAP_FRAMES,
        max_age_frames=TRACK_MAX_AGE_FRAMES,
    )

    while True:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed.")
            break

        frame_count += 1
        display = frame.copy()
        regions = []
        hsv_mask = None
        full_detections = []
        crop_detections = []
        full_scan_ran = False
        roi_scan_ran = False

        if hybrid_mode:
            # Regions always come from this frame; no ROI boxes/detections are
            # cached or reused after the camera moves.
            regions, hsv_mask = detect_colored_regions(frame)

            full_scan_ran = (
                (frame_count - 1) % HYBRID_FULL_EVERY_N_FRAMES == 0
            )
            roi_scan_ran = bool(regions) and (
                (frame_count - 1) % HYBRID_ROI_EVERY_N_FRAMES == 0
            )

            if full_scan_ran:
                full_detections = run_yolo_full_frame(
                    model, frame, full_confidence
                )
            if roi_scan_ran:
                crop_detections = run_yolo_on_regions(
                    model, frame, regions, crop_confidence
                )
        else:
            # FULL mode is a clean baseline: no HSV preprocessing or crops.
            full_scan_ran = True
            full_detections = run_yolo_full_frame(
                model, frame, full_confidence
            )

        merged = merge_detections(full_detections, crop_detections)
        tracked = confirmer.update(merged, frame_count)
        confirmed = [
            detection for detection in tracked if detection["confirmed"]
        ]

        for detection in tracked:
            draw_detection(display, detection, show_unconfirmed)

        for detection in tracked:
            if detection["newly_confirmed"]:
                detection_history.append(detection["cls_name"])
        detection_history = detection_history[-5:]

        # HSV overlays are debug information, not sign detections.
        if hybrid_mode and show_mask and hsv_mask is not None:
            mask_colored = np.zeros_like(display)
            mask_colored[:, :, 1] = hsv_mask
            display = cv2.addWeighted(display, 0.78, mask_colored, 0.22, 0)
            for x1, y1, x2, y2 in regions:
                cv2.rectangle(
                    display, (x1, y1), (x2, y2), (0, 200, 255), 1
                )
                cv2.putText(
                    display,
                    "ROI proposal",
                    (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 200, 255),
                    1,
                )

        processing_time = time.perf_counter() - loop_start
        instantaneous_fps = 1.0 / max(processing_time, 1e-6)
        fps_ema = (
            instantaneous_fps
            if fps_ema == 0.0
            else (0.90 * fps_ema) + (0.10 * instantaneous_fps)
        )

        if hybrid_mode:
            mode_text = "HYBRID: ROI + periodic full"
            scan_text = (
                f"full={'Y' if full_scan_ran else '-'} "
                f"roi={len(regions) if roi_scan_ran else 0}"
            )
        else:
            mode_text = "FULL: YOLO every frame"
            scan_text = "full=Y roi=0"

        info_text = (
            f"FPS {fps_ema:.1f} | YOLO {len(merged)} | "
            f"Confirmed {len(confirmed)} | {mode_text} | {scan_text}"
        )
        cv2.rectangle(
            display, (0, 0), (display.shape[1], 36), (0, 0, 0), -1
        )
        cv2.putText(
            display,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
        )

        sidebar_width = 235
        sidebar_x = display.shape[1] - sidebar_width
        sidebar_height = 34 + (len(detection_history) * 23)
        cv2.rectangle(
            display,
            (sidebar_x, 40),
            (display.shape[1], 40 + sidebar_height),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            display,
            "Confirmed history:",
            (sidebar_x + 7, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 0),
            1,
        )
        for index, name in enumerate(detection_history):
            cv2.putText(
                display,
                f"> {name}",
                (sidebar_x + 7, 84 + (index * 23)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (210, 210, 210),
                1,
            )

        controls = (
            "q Quit | s Screenshot | m HSV/ROI | h Mode | "
            "u Candidates | [ ] Confidence"
        )
        frame_h = display.shape[0]
        cv2.rectangle(
            display, (0, frame_h - 30), (display.shape[1], frame_h), (0, 0, 0), -1
        )
        cv2.putText(
            display,
            controls,
            (10, frame_h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (205, 205, 205),
            1,
        )

        cv2.imshow("MYSignVoice - Safe Hybrid Detection", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break
        if key == ord("s"):
            screenshot_count += 1
            mode_slug = "hybrid" if hybrid_mode else "full"
            save_path = os.path.join(
                SCREENSHOT_DIR,
                f"{mode_slug}_screenshot_{screenshot_count}.png",
            )
            cv2.imwrite(save_path, display)
            print(f"Screenshot saved: {save_path}")
        elif key == ord("m"):
            show_mask = not show_mask
            print(f"HSV/ROI debug overlay: {'ON' if show_mask else 'OFF'}")
        elif key == ord("h"):
            hybrid_mode = not hybrid_mode
            confirmer.reset()
            mode_name = "HYBRID" if hybrid_mode else "FULL"
            print(f"Mode switched to {mode_name}; confirmation tracks reset.")
        elif key == ord("u"):
            show_unconfirmed = not show_unconfirmed
            print(
                "Unconfirmed YOLO candidates: "
                f"{'VISIBLE' if show_unconfirmed else 'HIDDEN'}"
            )
        elif key == ord("["):
            full_confidence = max(
                0.05, full_confidence - CONFIDENCE_STEP
            )
            crop_confidence = max(
                0.05, crop_confidence - CONFIDENCE_STEP
            )
            confirmer.reset()
            print(
                f"Confidence: full={full_confidence:.2f}, "
                f"crop={crop_confidence:.2f}; tracks reset."
            )
        elif key == ord("]"):
            full_confidence = min(
                0.95, full_confidence + CONFIDENCE_STEP
            )
            crop_confidence = min(
                0.95, crop_confidence + CONFIDENCE_STEP
            )
            confirmer.reset()
            print(
                f"Confidence: full={full_confidence:.2f}, "
                f"crop={crop_confidence:.2f}; tracks reset."
            )

    elapsed = time.perf_counter() - start_time
    cap.release()
    cv2.destroyAllWindows()

    print(f"Done. Processed {frame_count} frames.")
    if elapsed > 0:
        print(f"Session average: {frame_count / elapsed:.1f} FPS.")
    if screenshot_count:
        print(f"Screenshots saved under: {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
