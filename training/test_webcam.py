"""Simple full-frame webcam smoke test for a trained YOLO26 detector.

Ultralytics applies aspect-preserving letterbox preprocessing internally. For
the production color-ROI, periodic-full-frame, and temporal-confirmation path,
use preliminary/member4_shape_detection/webcam_yolo_demo.py instead.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "best.pt"
EXPECTED_CLASS_COUNT = 63


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained YOLO26 weights or export.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--conf", type=float, default=0.4, help="Minimum detection confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cpu or 0. Omit for Ultralytics auto-selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")
    if args.imgsz <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("--imgsz, --width, and --height must be greater than zero")

    model = YOLO(args.model, task="detect")
    if len(model.names) != EXPECTED_CLASS_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_CLASS_COUNT} model classes, found {len(model.names)}"
        )
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    predict_options = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_options["device"] = args.device

    print("Webcam smoke test started. Press q or Esc to quit.")
    print("Use preliminary/member4_shape_detection/webcam_yolo_demo.py for production-pipeline testing.")
    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame capture failed.")
                break

            result = model.predict(source=frame, **predict_options)[0]
            annotated = result.plot()

            current_time = time.perf_counter()
            instantaneous_fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            smoothed_fps = instantaneous_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            cv2.putText(
                annotated,
                f"FPS {smoothed_fps:.1f} | inference {result.speed.get('inference', 0.0):.1f} ms",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLO26 traffic-sign webcam smoke test", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
