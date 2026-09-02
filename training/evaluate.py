"""Validate a trained YOLO26 detector on a labeled dataset split.

Ultralytics calculates object-detection precision, recall, mAP50, and
mAP50-95 and writes report-ready plots, including a confusion matrix. Plain
classification "accuracy" is intentionally not reported for this detection
task. Use benchmark_pipeline_modes.py separately for end-to-end latency/FPS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "best.pt"
DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "results" / "validation"
EXPECTED_CLASS_COUNT = 63


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained YOLO26 weights or export.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Ultralytics dataset YAML.")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Labeled dataset split.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Dataset worker processes.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Validation output root.")
    parser.add_argument("--name", default="yolo26s_640", help="Validation run name.")
    parser.add_argument(
        "--device",
        default=None,
        help="Validation device, for example cpu or 0. Omit for Ultralytics auto-selection.",
    )
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def ordered_names(raw_names: Any) -> list[str]:
    """Normalize list- or dictionary-style Ultralytics class names."""
    if isinstance(raw_names, dict):
        return [
            str(raw_names.get(index, raw_names.get(str(index))))
            for index in range(len(raw_names))
        ]
    return [str(name) for name in raw_names]


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    if args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise SystemExit("--imgsz and --batch must be positive; --workers cannot be negative")

    model = YOLO(args.model, task="detect")
    if len(model.names) != EXPECTED_CLASS_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_CLASS_COUNT} model classes, found {len(model.names)}"
        )
    data_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    model_names = ordered_names(model.names)
    data_names = ordered_names(data_config.get("names", []))
    if model_names != data_names:
        raise SystemExit(
            "Model and dataset class-ID orders differ. Evaluate only against the "
            "exact Roboflow export used by this model; do not remap IDs silently."
        )
    validation_options = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }
    if args.device:
        validation_options["device"] = args.device

    metrics = model.val(**validation_options)
    summary = {
        "model": args.model,
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "results": json_value(metrics.results_dict),
        "speed_ms_per_image": json_value(metrics.speed),
    }
    save_dir = Path(metrics.save_dir)
    summary_path = save_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nValidation complete.")
    for metric_name, metric_value in metrics.results_dict.items():
        print(f"  {metric_name}: {float(metric_value):.6f}")
    print(f"Plots and summary: {save_dir.resolve()}")
    print("Measure full webcam-pipeline latency separately with benchmark_pipeline_modes.py.")


if __name__ == "__main__":
    main()
