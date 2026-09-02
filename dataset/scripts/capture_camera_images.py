"""Collect unannotated traffic-sign images from a webcam for Roboflow.

Controls:
    S       save the current clean camera frame
    Q/Esc   close the camera window
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "raw_images" / "camera_capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a webcam and save a clean image whenever S is pressed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where captured images are saved.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.camera < 0:
        raise SystemExit("--camera must be zero or greater.")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be greater than zero.")

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Close other camera apps or try --camera 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    saved_count = 0
    window_name = "MYSignVoice Data Collection"
    print(f"Saving clean camera images to: {output_dir}")
    print("Press S to save an image. Press Q or Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a camera frame.")
                break

            # Draw instructions on a copy so saved images contain no overlay text.
            preview = frame.copy()
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 42), (0, 0, 0), -1)
            cv2.putText(
                preview,
                f"S: save | Q/Esc: quit | saved: {saved_count}",
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = output_dir / f"camera_{timestamp}.jpg"
                if not cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    print(f"Failed to save: {save_path}")
                    continue
                saved_count += 1
                print(f"Saved {saved_count}: {save_path.name}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Finished. Saved {saved_count} image(s) in {output_dir}")


if __name__ == "__main__":
    main()
