"""Train MYSignVoice's verified 63-class dataset in Google Colab.

Run after mounting Google Drive and installing the current project dependencies:

    !pip install "ultralytics>=8.4.90,<9" roboflow pyyaml openvino onnx onnxruntime
    !python train_colab.py --api-key "$ROBOFLOW_API_KEY" \
      --workspace "your-workspace" --project "mysignvoice-49-signs" --version 1

The script downloads the selected Roboflow version to Colab's local disk, checks
that its class names and order exactly match dataset/data.yaml, trains YOLO26s
by default, exports ONNX and OpenVINO models for web/Intel CPU deployment, and
then copies the complete run to Google Drive.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO


EXPECTED_CLASSES = [
    "straight-or-right", "straight-only", "left-turn-only",
    "left-or-right", "right-turn-only", "pass-right", "roundabout", "cars-only",
    "use-horn", "bicycle-path", "uturn-lane", "speed-limit-5", "speed-limit-15",
    "speed-limit-30", "speed-limit-40", "speed-limit-50", "speed-limit-60",
    "speed-limit-80", "no-straight-or-left", "no-straight", "no-left",
    "no-left-and-right", "no-right", "no-overtaking", "no-uturn", "no-cars",
    "no-horn", "traffic-light-ahead", "stop-sign", "no-entry", "give-way",
    "stop-for-inspection", "pass-obstacle-on-either-side", "general-warning",
    "pedestrian-crossing-warning", "bicycle-warning", "children-crossing-warning",
    "sharp-right-turn-warning", "steep-descent-warning", "slowdown-warning",
    "village-ahead-warning", "winding-road-warning",
    "railway-crossing-ahead-warning", "construction-ahead-warning",
    "slippery-road-warning", "gated-railway-crossing-ahead-warning",
    "accident-prone-area-warning",
    "bumps-warning", "bus-stop", "camera-operation-zone",
    "cow-nearby-warning", "height-limit", "no-parking",
    "parking-area", "towing-area", "chevron-left", "chevron-right",
    "crossroad-left-warning", "crossroad-right-warning",
    "road-narrows-left-warning", "road-narrows-right-warning",
    "roadway-diverges-warning", "reverse-turn-warning",
]

CLASS_NAME_CORRECTIONS = {
    "pass-obstacles-on-either-side": "pass-obstacle-on-either-side",
    "winding_road_warning": "winding-road-warning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"),
                        help="Roboflow API key. Defaults to ROBOFLOW_API_KEY.")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug.")
    parser.add_argument("--project", required=True, help="Roboflow project slug.")
    parser.add_argument("--version", required=True, type=int, help="Generated Roboflow version.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/content/mysignvoice_dataset"))
    parser.add_argument("--run-dir", type=Path, default=Path("/content/mysignvoice_runs"))
    parser.add_argument("--drive-dir", type=Path,
                        default=Path("/content/drive/MyDrive/TrafficSignProject/training_runs"))
    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="Pretrained Ultralytics detection weights (default: yolo26s.pt).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to a model/version-specific name.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def normalise_names(names: list[str] | dict) -> list[str]:
    """Return YOLO names as an ordered list, accepting YAML list or ID dictionary."""
    if isinstance(names, list):
        return names
    if isinstance(names, dict):
        return [names[index] if index in names else names[str(index)] for index in range(len(names))]
    raise ValueError("data.yaml has no usable 'names' list or dictionary.")


def validate_dataset_config(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    original_names = normalise_names(config.get("names", []))
    names = [CLASS_NAME_CORRECTIONS.get(name, name) for name in original_names]
    missing = sorted(set(EXPECTED_CLASSES) - set(names))
    unexpected = sorted(set(names) - set(EXPECTED_CLASSES))
    if (
        config.get("nc") != len(EXPECTED_CLASSES)
        or len(names) != len(EXPECTED_CLASSES)
        or len(set(names)) != len(EXPECTED_CLASSES)
        or missing
        or unexpected
    ):
        actual = "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
        expected = "\n".join(f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASSES))
        raise ValueError(
            "Roboflow's exported class set does not match the locked 63-class inventory.\n"
            f"Missing: {missing}\nUnexpected: {unexpected}\n\n"
            f"Expected inventory:\n{expected}\n\nExported order:\n{actual}"
        )

    if names != original_names:
        config["names"] = names
        config["nc"] = len(names)
        with data_yaml.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        print("Corrected known class-name variants in the downloaded data.yaml.")

    print("Class-list check passed: all 63 class names are present and unique.")
    print("Roboflow's exported class-ID order is preserved for labels and model output.")
    return names


def report_label_counts(dataset_root: Path, class_names: list[str]) -> None:
    """Show per-class label counts and flag classes unsuitable for meaningful metrics."""
    counts: Counter[int] = Counter()
    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "val", "test"):
        label_dir = dataset_root / split / "labels"
        if not label_dir.is_dir():
            continue
        label_files = list(label_dir.glob("*.txt"))
        split_counts[split] = len(label_files)
        for label_file in label_files:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1

    print(f"Split label files: {split_counts}")
    scarce = []
    for class_id, class_name in enumerate(class_names):
        count = counts[class_id]
        print(f"{class_id:>2}  {class_name:<42} {count:>4} boxes")
        if count < 5:
            scarce.append(class_name)
    if scarce:
        print("\nWARNING: Classes with fewer than 5 labelled boxes:")
        print(", ".join(scarce))
        print("Collect more original photographs before treating validation/test metrics as reliable.")


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set ROBOFLOW_API_KEY.")

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit(
            "The Roboflow package is required. In Colab run: pip install roboflow"
        ) from exc

    model_stem = Path(args.model).stem
    run_name = args.run_name or f"{model_stem}_63class_rf_v{args.version}"

    rf = Roboflow(api_key=args.api_key)
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    dataset = version.download(model_format="yolov8", location=str(args.dataset_dir))
    dataset_root = Path(dataset.location)
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Roboflow export did not contain data.yaml: {data_yaml}")

    exported_class_names = validate_dataset_config(data_yaml)
    report_label_counts(dataset_root, exported_class_names)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=30,
        project=str(args.run_dir),
        name=run_name,
        exist_ok=True,
        cache="disk",
        plots=True,
        # Geometric/colour augmentation is safe; horizontal flipping is not because
        # it changes the meaning of left/right/turn traffic signs.
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,
        degrees=8,
        translate=0.05,
        scale=0.25,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.50,
        close_mosaic=10,
    )

    local_run = args.run_dir / run_name
    best_model = local_run / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"Training finished without best.pt at {best_model}")

    # Export before copying so Drive contains the PyTorch checkpoint plus portable
    # ONNX and Intel-CPU-friendly OpenVINO artifacts for the web backend.
    export_model = YOLO(str(best_model))
    onnx_export = Path(
        export_model.export(
            format="onnx", imgsz=args.imgsz, simplify=True, end2end=True
        )
    )
    openvino_export = Path(
        export_model.export(format="openvino", imgsz=args.imgsz, end2end=True)
    )
    if not onnx_export.is_file():
        raise FileNotFoundError(f"ONNX export was not created: {onnx_export}")
    if not openvino_export.is_dir():
        raise FileNotFoundError(f"OpenVINO export was not created: {openvino_export}")

    drive_run = args.drive_dir / run_name
    drive_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_run, drive_run, dirs_exist_ok=True)
    print(f"\nTraining complete. Full run copied to: {drive_run}")
    print(f"Best PyTorch model: {drive_run / 'weights' / 'best.pt'}")
    print(f"ONNX web model: {drive_run / 'weights' / onnx_export.name}")
    print(f"OpenVINO Intel CPU model: {drive_run / 'weights' / openvino_export.name}")


if __name__ == "__main__":
    main()
