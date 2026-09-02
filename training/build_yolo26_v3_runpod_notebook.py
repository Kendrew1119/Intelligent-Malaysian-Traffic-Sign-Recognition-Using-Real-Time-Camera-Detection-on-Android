"""Build the self-contained MYSignVoice Version 3 RunPod notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "train_yolo26_v3_runpod.ipynb"


def lines(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        """
        # MYSignVoice V3: YOLO26s fine-tuning on RunPod

        This notebook fine-tunes the deployed 63-class Version 2 checkpoint on
        Roboflow Version 3. It uses all old data plus the 367 newly annotated
        laptop-camera images. It does **not** start again from a generic model.

        Run every cell from top to bottom. In the default API mode, upload only:

        - `best_v2.pt` — the currently deployed Version 2 checkpoint

        Cell 3 securely asks for the Roboflow private API key and downloads
        Version 3 directly to RunPod. The key is hidden and is not saved in the
        notebook. A manually uploaded `mysignvoice_v3_yolo26.zip` is optional.

        The notebook keeps the held-out Test split separate, compares V2 and V3
        on Validation, exports deployment files, and creates one downloadable ZIP.
        """
    ),
    markdown("## Cell 1 — Install the tested packages"),
    code(
        """
        %pip install -q "ultralytics==8.4.128" roboflow pyyaml pandas matplotlib seaborn openvino onnx onnxruntime
        """
    ),
    markdown("## Cell 2 — Confirm the GPU and uploaded files"),
    code(
        """
        import json
        import os
        import platform
        import random
        import re
        import shutil
        import zipfile
        from collections import Counter, defaultdict
        from datetime import datetime
        from getpass import getpass
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import ultralytics
        import yaml
        from IPython.display import FileLink, display
        from ultralytics import YOLO

        WORKSPACE_ROOT = Path("/workspace")
        DATASET_ZIP = WORKSPACE_ROOT / "mysignvoice_v3_yolo26.zip"
        BASE_MODEL_PATH = WORKSPACE_ROOT / "best_v2.pt"

        VERSION = 3
        WORKSPACE = "kendrewlim-yahoo-com"
        PROJECT = "mysignvoice-49-signs"
        USE_ROBOFLOW_API = True
        EXPECTED_CLASSES = 63
        EXPECTED_NEW_CAMERA_IMAGES = 367
        CAMERA_PREFIX = "camera_20260901_"

        IMAGE_SIZE = 640
        EPOCHS = 60
        BATCH_SIZE = 16
        PATIENCE = 15
        WORKERS = 8
        CLASS_WEIGHT_POWER = 0.25
        RUN_FINAL_TEST = False

        ARCHIVE_ROOT = WORKSPACE_ROOT / "mysignvoice_rf_v3"
        DATASET_ROOT = ARCHIVE_ROOT
        RUN_ROOT = WORKSPACE_ROOT / "mysignvoice_runs"
        RUN_NAME = "yolo26s_63class_rf_v3_camera_finetune"
        RUN_DIR = RUN_ROOT / RUN_NAME
        EVAL_ROOT = WORKSPACE_ROOT / "mysignvoice_evaluations_v3"
        REPORT_DIR = WORKSPACE_ROOT / "mysignvoice_report_v3"
        PACKAGE_ROOT = WORKSPACE_ROOT / "mysignvoice_packages"
        for folder in (RUN_ROOT, EVAL_ROOT, REPORT_DIR, PACKAGE_ROOT):
            folder.mkdir(parents=True, exist_ok=True)

        assert BASE_MODEL_PATH.is_file(), f"Upload and rename the current checkpoint to: {BASE_MODEL_PATH}"
        assert torch.cuda.is_available(), "No CUDA GPU found. Start a RunPod GPU pod, not a CPU-only pod."

        print("Python:", platform.python_version())
        print("Ultralytics:", ultralytics.__version__)
        print("PyTorch:", torch.__version__)
        print("GPU:", torch.cuda.get_device_name(0))
        print("Dataset source:", "Roboflow API" if USE_ROBOFLOW_API else DATASET_ZIP)
        print("Starting checkpoint:", BASE_MODEL_PATH)
        """
    ),
    markdown("## Cell 3 — Extract Version 3 and verify the 63-class ID order"),
    code(
        """
        if USE_ROBOFLOW_API:
            existing_api_yaml = sorted(ARCHIVE_ROOT.rglob("data.yaml")) if ARCHIVE_ROOT.exists() else []
            if existing_api_yaml:
                print("Reusing the existing Roboflow Version 3 download:", ARCHIVE_ROOT)
            else:
                if ARCHIVE_ROOT.exists():
                    shutil.rmtree(ARCHIVE_ROOT)
                from roboflow import Roboflow

                api_key = getpass("Paste the Roboflow private API key (hidden): ")
                assert api_key.strip(), "No Roboflow API key was entered."
                rf = Roboflow(api_key=api_key.strip())
                version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
                dataset = version.download(model_format="yolo26", location=str(ARCHIVE_ROOT))
                del api_key
                print("Roboflow download complete:", dataset.location)
        else:
            EXPECTED_ZIP_BYTES = 276_547_196
            assert DATASET_ZIP.is_file(), f"Missing dataset ZIP: {DATASET_ZIP}"
            actual_zip_bytes = DATASET_ZIP.stat().st_size
            print(f"Uploaded ZIP: {actual_zip_bytes:,} bytes")
            assert zipfile.is_zipfile(DATASET_ZIP), (
                "The uploaded dataset is incomplete or is not a ZIP file. "
                f"Expected the verified V3 export ({EXPECTED_ZIP_BYTES:,} bytes)."
            )
            extraction_marker = ARCHIVE_ROOT / ".extraction_complete"
            if not extraction_marker.is_file():
                if ARCHIVE_ROOT.exists():
                    shutil.rmtree(ARCHIVE_ROOT)
                ARCHIVE_ROOT.mkdir(parents=True)
                with zipfile.ZipFile(DATASET_ZIP) as archive:
                    archive.extractall(ARCHIVE_ROOT)
                extraction_marker.write_text("complete", encoding="utf-8")

        yaml_candidates = sorted(ARCHIVE_ROOT.rglob("data.yaml"))
        assert len(yaml_candidates) == 1, f"Expected one data.yaml, found: {yaml_candidates}"
        DATA_YAML = yaml_candidates[0]
        DATASET_ROOT = DATA_YAML.parent

        with DATA_YAML.open("r", encoding="utf-8") as stream:
            data_config = yaml.safe_load(stream)

        exported_names = data_config.get("names", [])
        if isinstance(exported_names, dict):
            exported_names = [
                exported_names[i] if i in exported_names else exported_names[str(i)]
                for i in range(len(exported_names))
            ]

        NAME_CORRECTIONS = {
            "pass-obstacles-on-either-side": "pass-obstacle-on-either-side",
            "winding_road_warning": "winding-road-warning",
        }
        exported_names = [NAME_CORRECTIONS.get(name, name) for name in exported_names]
        data_config["names"] = exported_names
        data_config["nc"] = len(exported_names)
        DATA_YAML.write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")

        base_model = YOLO(str(BASE_MODEL_PATH), task="detect")
        deployed_names = base_model.names
        if isinstance(deployed_names, dict):
            deployed_names = [
                deployed_names[i] if i in deployed_names else deployed_names[str(i)]
                for i in range(len(deployed_names))
            ]
        deployed_names = [NAME_CORRECTIONS.get(name, name) for name in deployed_names]

        assert len(exported_names) == EXPECTED_CLASSES
        assert len(set(exported_names)) == EXPECTED_CLASSES
        assert exported_names == deployed_names, (
            "STOP: Version 3 class IDs do not match best_v2.pt.\\n"
            f"Checkpoint: {deployed_names}\\nExport: {exported_names}"
        )
        CLASS_NAMES = exported_names
        print("PASS: Version 3 and best_v2.pt contain the same 63 class IDs.")
        print("Dataset root:", DATASET_ROOT)
        """
    ),
    markdown(
        """
        ## Cell 4 — Regroup the 367 consecutive camera frames

        Roboflow assigned the new batch to Train, Validation and Test, but
        neighbouring screenshots can be almost identical. This cell groups
        frames with the same labels and no more than eight seconds between them,
        then puts each whole group in one split. Old Version 2 splits are not moved.
        """
    ),
    code(
        """
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def resolve_split(yaml_key, fallback_folder):
            candidates = []
            raw_path = data_config.get(yaml_key)
            if raw_path:
                raw_path = Path(raw_path)
                candidates.append(raw_path if raw_path.is_absolute() else (DATA_YAML.parent / raw_path).resolve())
            candidates.append((DATASET_ROOT / fallback_folder / "images").resolve())
            image_dir = next((path for path in candidates if path.is_dir()), None)
            assert image_dir is not None, f"Cannot find {yaml_key} images. Tried: {candidates}"
            return image_dir

        split_image_dirs = {
            "train": resolve_split("train", "train"),
            "val": resolve_split("val", "valid"),
            "test": resolve_split("test", "test"),
        }

        camera_pattern = re.compile(r"camera_(\\d{8})_(\\d{6})_(\\d{6})")

        def capture_time(path):
            match = camera_pattern.search(path.stem)
            assert match, f"Cannot parse camera timestamp: {path.name}"
            return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S%f")

        def class_signature(label_path):
            if not label_path.is_file():
                return tuple()
            ids = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.append(int(line.split()[0]))
            return tuple(sorted(ids))

        records = []
        for split_name, image_dir in split_image_dirs.items():
            label_dir = image_dir.parent / "labels"
            for image_path in image_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.stem.startswith(CAMERA_PREFIX):
                    continue
                label_path = label_dir / f"{image_path.stem}.txt"
                records.append({
                    "image": image_path,
                    "label": label_path,
                    "original_split": split_name,
                    "timestamp": capture_time(image_path),
                    "signature": class_signature(label_path),
                })

        assert len(records) == EXPECTED_NEW_CAMERA_IMAGES, (
            f"Expected {EXPECTED_NEW_CAMERA_IMAGES} new camera images, found {len(records)}. "
            "Confirm you exported Roboflow Version 3."
        )

        records.sort(key=lambda item: item["timestamp"])
        groups = []
        current_group = []
        for record in records:
            if current_group:
                gap = (record["timestamp"] - current_group[-1]["timestamp"]).total_seconds()
                label_changed = record["signature"] != current_group[-1]["signature"]
                if gap > 8 or label_changed:
                    groups.append(current_group)
                    current_group = []
            current_group.append(record)
        if current_group:
            groups.append(current_group)

        rng = random.Random(42)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        target_ratio = {"train": 0.71, "val": 0.14, "test": 0.15}
        target_count = {name: ratio * len(records) for name, ratio in target_ratio.items()}
        assigned_count = {name: 0 for name in target_ratio}
        group_assignments = []
        split_priority = {"train": 0, "val": 1, "test": 2}

        for group_id, group in enumerate(groups):
            target_split = min(
                target_ratio,
                key=lambda name: (
                    assigned_count[name] / max(target_count[name], 1),
                    split_priority[name],
                ),
            )
            assigned_count[target_split] += len(group)
            group_assignments.append((group_id, target_split, group))

        stage_root = DATASET_ROOT / "_camera_grouped_stage"
        if stage_root.exists():
            shutil.rmtree(stage_root)
        (stage_root / "images").mkdir(parents=True)
        (stage_root / "labels").mkdir(parents=True)

        manifest_rows = []
        for group_id, target_split, group in group_assignments:
            for record in group:
                staged_image = stage_root / "images" / record["image"].name
                shutil.copy2(record["image"], staged_image)
                staged_label = stage_root / "labels" / record["label"].name
                if record["label"].is_file():
                    shutil.copy2(record["label"], staged_label)
                manifest_rows.append({
                    "image": record["image"].name,
                    "group_id": group_id,
                    "original_split": record["original_split"],
                    "assigned_split": target_split,
                    "class_ids": ",".join(map(str, record["signature"])),
                    "timestamp": record["timestamp"].isoformat(),
                })

        assert len(list((stage_root / "images").iterdir())) == len(records)
        for record in records:
            record["image"].unlink()
            if record["label"].is_file():
                record["label"].unlink()

        for row in manifest_rows:
            target_image_dir = split_image_dirs[row["assigned_split"]]
            target_label_dir = target_image_dir.parent / "labels"
            target_label_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage_root / "images" / row["image"]), target_image_dir / row["image"])
            staged_label = stage_root / "labels" / f"{Path(row['image']).stem}.txt"
            if staged_label.is_file():
                shutil.move(str(staged_label), target_label_dir / staged_label.name)
        shutil.rmtree(stage_root)

        camera_split_manifest = pd.DataFrame(manifest_rows)
        camera_split_manifest.to_csv(REPORT_DIR / "camera_group_split_manifest.csv", index=False)
        leakage = camera_split_manifest.groupby("group_id")["assigned_split"].nunique()
        assert int(leakage.max()) == 1, "A camera group leaked across splits."
        print("Camera groups:", len(groups))
        print("Leakage-safe new-image split:", camera_split_manifest["assigned_split"].value_counts().to_dict())
        print("PASS: each consecutive camera group belongs to only one split.")
        """
    ),
    markdown("## Cell 5 — Audit labels, duplicates, splits and class balance"),
    code(
        """
        import hashlib

        # Remove only cross-split copies when identical pixels have the same class IDs.
        # Hold-out data has priority: Test > Validation > Train.
        duplicate_records = defaultdict(list)
        for split_name, image_dir in split_image_dirs.items():
            label_dir = image_dir.parent / "labels"
            for image_path in image_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label_file = label_dir / f"{image_path.stem}.txt"
                class_ids = tuple()
                if label_file.is_file():
                    class_ids = tuple(sorted(
                        int(line.split()[0])
                        for line in label_file.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ))
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                duplicate_records[digest].append({
                    "split": split_name,
                    "image": image_path,
                    "label": label_file,
                    "class_ids": class_ids,
                })

        cross_split_records = {
            digest: entries
            for digest, entries in duplicate_records.items()
            if len({entry["split"] for entry in entries}) > 1
        }
        class_conflicts = {
            digest: entries
            for digest, entries in cross_split_records.items()
            if len({entry["class_ids"] for entry in entries}) > 1
        }
        assert not class_conflicts, (
            "STOP: identical pixels have conflicting class IDs across splits. "
            f"Conflicting hashes: {list(class_conflicts)[:10]}"
        )

        split_priority = {"train": 0, "val": 1, "test": 2}
        duplicate_removal_rows = []
        for digest, entries in cross_split_records.items():
            keeper_split = max(
                {entry["split"] for entry in entries},
                key=lambda name: split_priority[name],
            )
            for entry in entries:
                if entry["split"] == keeper_split:
                    continue
                entry["image"].unlink()
                if entry["label"].is_file():
                    entry["label"].unlink()
                duplicate_removal_rows.append({
                    "sha256": digest,
                    "removed_split": entry["split"],
                    "kept_split": keeper_split,
                    "removed_image": entry["image"].name,
                    "class_ids": ",".join(map(str, entry["class_ids"])),
                })

        pd.DataFrame(duplicate_removal_rows).to_csv(
            REPORT_DIR / "cross_split_duplicate_removals.csv", index=False
        )
        print(
            "Removed exact cross-split duplicate copies:",
            len(duplicate_removal_rows),
            "(held-out split retained)",
        )

        split_rows = []
        box_counts = {split: Counter() for split in split_image_dirs}
        invalid_labels = []
        image_hash_splits = defaultdict(set)

        for split_name, image_dir in split_image_dirs.items():
            label_dir = image_dir.parent / "labels"
            images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
            assert images, f"The {split_name} split has no images."
            images_with_boxes = 0
            total_boxes = 0
            for image_path in images:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                image_hash_splits[digest].add(split_name)
                label_file = label_dir / f"{image_path.stem}.txt"
                image_box_count = 0
                if label_file.is_file():
                    for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
                        if not line.strip():
                            continue
                        parts = line.split()
                        try:
                            class_id = int(parts[0])
                            coordinates = [float(value) for value in parts[1:]]
                            assert len(coordinates) == 4
                            assert 0 <= class_id < EXPECTED_CLASSES
                            assert all(0.0 <= value <= 1.0 for value in coordinates)
                            assert coordinates[2] > 0.0 and coordinates[3] > 0.0
                        except (ValueError, AssertionError):
                            invalid_labels.append(f"{label_file}:{line_number}: {line}")
                            continue
                        image_box_count += 1
                        total_boxes += 1
                        box_counts[split_name][class_id] += 1
                if image_box_count:
                    images_with_boxes += 1
            split_rows.append({
                "split": split_name,
                "images": len(images),
                "images_with_boxes": images_with_boxes,
                "negative_images": len(images) - images_with_boxes,
                "boxes": total_boxes,
            })

        assert not invalid_labels, "Invalid labels:\\n" + "\\n".join(invalid_labels[:20])
        duplicate_splits = {
            digest: sorted(splits)
            for digest, splits in image_hash_splits.items()
            if len(splits) > 1
        }
        assert not duplicate_splits, (
            "STOP: exact duplicate image content exists across splits: "
            + json.dumps(dict(list(duplicate_splits.items())[:10]), indent=2)
        )

        split_summary = pd.DataFrame(split_rows).set_index("split")
        ACTUAL_SPLIT_IMAGES = {name: int(split_summary.loc[name, "images"]) for name in split_image_dirs}
        missing_train_classes = [CLASS_NAMES[i] for i in range(EXPECTED_CLASSES) if box_counts["train"][i] == 0]
        assert not missing_train_classes, f"STOP: classes missing from Train: {missing_train_classes}"

        class_rows = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_rows.append({
                "class_id": class_id,
                "class_name": class_name,
                "train_boxes": box_counts["train"][class_id],
                "val_boxes": box_counts["val"][class_id],
                "test_boxes": box_counts["test"][class_id],
            })
        class_distribution = pd.DataFrame(class_rows)
        split_summary.to_csv(REPORT_DIR / "dataset_split_summary.csv")
        class_distribution.to_csv(REPORT_DIR / "class_distribution.csv", index=False)
        (REPORT_DIR / "split_counts.json").write_text(
            json.dumps(ACTUAL_SPLIT_IMAGES, indent=2), encoding="utf-8"
        )

        ax = split_summary["images"].plot(kind="bar", figsize=(8, 5), color=["#2563eb", "#14b8a6", "#f59e0b"])
        ax.set_title("MYSignVoice Version 3 dataset split")
        ax.set_ylabel("Images")
        ax.set_xlabel("")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "dataset_split.png", dpi=200)
        plt.show()

        plot_data = class_distribution.sort_values("train_boxes")
        plt.figure(figsize=(12, 16))
        plt.barh(plot_data["class_name"], plot_data["train_boxes"], color="#2563eb")
        plt.title("Version 3 training boxes by class")
        plt.xlabel("Bounding boxes")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "class_distribution.png", dpi=200)
        plt.show()

        display(split_summary)
        display(class_distribution.sort_values(["train_boxes", "class_id"]).head(20))
        print("Final split after camera grouping:", ACTUAL_SPLIT_IMAGES)
        print("PASS: labels, class coverage, exact duplicates and camera-group leakage checks passed.")
        SPLIT_APPROVED = True
        """
    ),
    markdown("## Cell 6 — Measure the deployed V2 model on V3 Validation"),
    code(
        """
        assert SPLIT_APPROVED
        baseline_metrics = base_model.val(
            data=str(DATA_YAML), split="val", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
            device=0, project=str(EVAL_ROOT),
            name="deployed_v2_on_v3_validation", exist_ok=True, plots=True,
        )
        BASELINE_SUMMARY = {
            "model": "deployed Version 2 best.pt",
            "precision": float(baseline_metrics.box.mp),
            "recall": float(baseline_metrics.box.mr),
            "mAP50": float(baseline_metrics.box.map50),
            "mAP50-95": float(baseline_metrics.box.map),
        }
        (REPORT_DIR / "baseline_validation_summary.json").write_text(
            json.dumps(BASELINE_SUMMARY, indent=2), encoding="utf-8"
        )
        pd.DataFrame({
            "class_id": range(EXPECTED_CLASSES),
            "class_name": CLASS_NAMES,
            "baseline_mAP50-95": np.asarray(baseline_metrics.box.maps, dtype=float),
        }).to_csv(REPORT_DIR / "baseline_validation_per_class.csv", index=False)
        display(pd.DataFrame([BASELINE_SUMMARY]))
        """
    ),
    markdown(
        """
        ## Cell 7 — Fine-tune from V2 on the complete V3 dataset

        This is full 63-class fine-tuning, not training only on the 367 images.
        The lower learning rate protects knowledge already learned by Version 2.
        """
    ),
    code(
        """
        assert SPLIT_APPROVED
        assert not RUN_DIR.exists(), (
            f"Run folder already exists: {RUN_DIR}. "
            "If training was interrupted, use the recovery cell at the end."
        )
        update_model = YOLO(str(BASE_MODEL_PATH), task="detect")
        update_model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            patience=PATIENCE,
            device=0,
            workers=WORKERS,
            project=str(RUN_ROOT),
            name=RUN_NAME,
            exist_ok=False,
            pretrained=True,
            resume=False,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.01,
            cos_lr=True,
            weight_decay=0.0005,
            cls_pw=CLASS_WEIGHT_POWER,
            cache="disk",
            amp=True,
            plots=True,
            save_period=5,
            seed=42,
            deterministic=True,
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
        BEST_PT = RUN_DIR / "weights" / "best.pt"
        LAST_PT = RUN_DIR / "weights" / "last.pt"
        assert BEST_PT.is_file(), f"Training did not create {BEST_PT}"
        print("New best checkpoint:", BEST_PT)
        """
    ),
    markdown("## Cell 8 — Compare V2 and V3 on Validation"),
    code(
        """
        BEST_PT = RUN_DIR / "weights" / "best.pt"
        assert BEST_PT.is_file(), f"Missing {BEST_PT}"
        best_model = YOLO(str(BEST_PT), task="detect")
        update_metrics = best_model.val(
            data=str(DATA_YAML), split="val", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
            device=0, project=str(EVAL_ROOT),
            name="updated_v3_validation", exist_ok=True, plots=True,
        )
        UPDATE_SUMMARY = {
            "model": "Version 3 fine-tuned best.pt",
            "precision": float(update_metrics.box.mp),
            "recall": float(update_metrics.box.mr),
            "mAP50": float(update_metrics.box.map50),
            "mAP50-95": float(update_metrics.box.map),
        }
        comparison = pd.DataFrame([BASELINE_SUMMARY, UPDATE_SUMMARY])
        comparison.to_csv(REPORT_DIR / "validation_comparison.csv", index=False)

        per_class = pd.read_csv(REPORT_DIR / "baseline_validation_per_class.csv")
        per_class["v3_mAP50-95"] = np.asarray(update_metrics.box.maps, dtype=float)
        per_class["change"] = per_class["v3_mAP50-95"] - per_class["baseline_mAP50-95"]
        per_class.to_csv(REPORT_DIR / "validation_per_class_comparison.csv", index=False)

        ax = comparison.set_index("model")[["precision", "recall", "mAP50", "mAP50-95"]].T.plot(
            kind="bar", figsize=(11, 6), color=["#94a3b8", "#2563eb"]
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Deployed V2 vs fine-tuned V3 on Validation")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "validation_comparison.png", dpi=200)
        plt.show()

        display(comparison)
        print("Largest per-class improvements:")
        display(per_class.sort_values("change", ascending=False).head(15))
        print("Largest per-class decreases:")
        display(per_class.sort_values("change").head(15))
        print("Speed-limit classes:")
        display(per_class[per_class["class_name"].str.startswith("speed-limit-")])
        """
    ),
    markdown(
        """
        ## Cell 9 — Optional one-time final Test

        Leave `RUN_FINAL_TEST = False` while you may still change settings.
        After accepting the Validation comparison, change it to `True` in Cell 2,
        rerun Cell 2, then run this cell once.
        """
    ),
    code(
        """
        if not RUN_FINAL_TEST:
            print("Final Test skipped. This is correct while tuning is not frozen.")
        else:
            base_test = YOLO(str(BASE_MODEL_PATH), task="detect").val(
                data=str(DATA_YAML), split="test", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                device=0, project=str(EVAL_ROOT),
                name="deployed_v2_final_test", exist_ok=True, plots=True,
            )
            update_test = YOLO(str(BEST_PT), task="detect").val(
                data=str(DATA_YAML), split="test", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                device=0, project=str(EVAL_ROOT),
                name="updated_v3_final_test", exist_ok=True, plots=True,
            )
            final_test_comparison = pd.DataFrame([
                {
                    "model": "deployed Version 2",
                    "precision": float(base_test.box.mp),
                    "recall": float(base_test.box.mr),
                    "mAP50": float(base_test.box.map50),
                    "mAP50-95": float(base_test.box.map),
                },
                {
                    "model": "fine-tuned Version 3",
                    "precision": float(update_test.box.mp),
                    "recall": float(update_test.box.mr),
                    "mAP50": float(update_test.box.map50),
                    "mAP50-95": float(update_test.box.map),
                },
            ])
            final_test_comparison.to_csv(REPORT_DIR / "final_test_comparison.csv", index=False)
            display(final_test_comparison)
        """
    ),
    markdown("## Cell 10 — Export PyTorch, ONNX and OpenVINO and package the evidence"),
    code(
        """
        BEST_PT = RUN_DIR / "weights" / "best.pt"
        assert BEST_PT.is_file(), f"Missing {BEST_PT}"
        best_model = YOLO(str(BEST_PT), task="detect")

        onnx_path = Path(best_model.export(format="onnx", imgsz=IMAGE_SIZE, simplify=True, end2end=True))
        openvino_path = Path(best_model.export(format="openvino", imgsz=IMAGE_SIZE, end2end=True))
        assert onnx_path.is_file(), f"ONNX export missing: {onnx_path}"
        assert openvino_path.is_dir(), f"OpenVINO export missing: {openvino_path}"

        val_images = sorted(split_image_dirs["val"].iterdir())[:10]
        assert val_images
        openvino_model = YOLO(str(openvino_path), task="detect")
        smoke_results = openvino_model.predict(
            source=[str(path) for path in val_images], imgsz=IMAGE_SIZE,
            conf=0.25, device="cpu", verbose=False,
        )
        assert len(smoke_results) == len(val_images)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        PACKAGE_DIR = PACKAGE_ROOT / f"mysignvoice_v3_training_{stamp}"
        PACKAGE_DIR.mkdir(parents=True, exist_ok=False)
        shutil.copytree(RUN_DIR, PACKAGE_DIR / "training_run")
        shutil.copytree(REPORT_DIR, PACKAGE_DIR / "report_artifacts")
        shutil.copy2(DATA_YAML, PACKAGE_DIR / "data.yaml")

        manifest = {
            "project": "MYSignVoice",
            "model": "YOLO26s",
            "roboflow_version": VERSION,
            "source_checkpoint": "deployed Version 2 best.pt",
            "best_checkpoint": "training_run/weights/best.pt",
            "classes": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "epochs_requested": EPOCHS,
            "split_images": ACTUAL_SPLIT_IMAGES,
            "new_camera_images": EXPECTED_NEW_CAMERA_IMAGES,
            "camera_grouping": "same labels and <=8 second gap remain in one split",
            "validation": UPDATE_SUMMARY,
            "final_test_run": RUN_FINAL_TEST,
            "exports": {
                "onnx": onnx_path.name,
                "openvino": openvino_path.name,
            },
        }
        (PACKAGE_DIR / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        LATEST_ARCHIVE = Path(shutil.make_archive(str(PACKAGE_DIR), "zip", root_dir=PACKAGE_DIR))
        print("PASS: model exports loaded successfully.")
        print("Package:", PACKAGE_DIR)
        print("ZIP:", LATEST_ARCHIVE)
        """
    ),
    markdown("## Cell 11 — Download the complete result ZIP"),
    code(
        """
        archives = sorted(PACKAGE_ROOT.glob("mysignvoice_v3_training_*.zip"), key=lambda path: path.stat().st_mtime)
        assert archives, "No result ZIP exists. Run Cell 10 first."
        LATEST_ARCHIVE = archives[-1]
        print("ZIP size (GB):", round(LATEST_ARCHIVE.stat().st_size / 1024**3, 3))
        display(FileLink(str(LATEST_ARCHIVE)))
        """
    ),
    markdown("## Recovery only — resume an interrupted Cell 7"),
    code(
        """
        LAST_PT = RUN_DIR / "weights" / "last.pt"
        assert LAST_PT.is_file(), f"No interrupted checkpoint found: {LAST_PT}"
        YOLO(str(LAST_PT), task="detect").train(resume=True)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
