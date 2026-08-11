"""Archived mobile conversion entry point; NCNN is not an active project target.

The project now deploys YOLO26s through a server/web pipeline. Export ONNX and
OpenVINO models with training/export_model.py instead. This file is retained
only so older notes or commands fail with a clear migration message.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    print("NCNN/mobile conversion is archived and is not part of the active web/server project.")
    print("Run: python training/export_model.py --model models/best.pt")
    print("That command creates the supported ONNX and OpenVINO exports.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
