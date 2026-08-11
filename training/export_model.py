"""Export a trained YOLO26 detector for the server/web inference pipeline.

The active deployment targets are ONNX and OpenVINO. YOLO26's native
end-to-end (NMS-free) output is retained in both exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Trained YOLO26 .pt weights (an official model name is also accepted).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square export image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Export device, for example cpu or 0. Omit for Ultralytics auto-selection.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset override. Omit to use the Ultralytics default.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be greater than zero")

    model = YOLO(args.model)
    common = {
        "imgsz": args.imgsz,
        "end2end": True,
    }
    if args.device:
        common["device"] = args.device

    onnx_options = {
        **common,
        "format": "onnx",
        "simplify": not args.no_simplify,
    }
    if args.opset is not None:
        onnx_options["opset"] = args.opset

    print("Exporting end-to-end ONNX model...")
    onnx_path = model.export(**onnx_options)

    print("Exporting end-to-end OpenVINO model...")
    openvino_path = model.export(format="openvino", **common)

    print("\nExports complete:")
    print(f"  ONNX:     {onnx_path}")
    print(f"  OpenVINO: {openvino_path}")
    print("Use the OpenVINO export for the Intel CPU web backend and keep ONNX as a portable fallback.")


if __name__ == "__main__":
    main()
