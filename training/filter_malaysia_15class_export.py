"""Build a safe, labelled MYSignVoice import from a Malaysia-road-sign YOLO ZIP.

By default only the 15 approved Malaysian additions are retained.  The optional
original-overlap mode also retains source signs that map to the original
47-class MYSignVoice seed inventory.  In both modes, an image is excluded when it
contains any unapproved source class, preventing an unknown sign from silently
becoming background in the target project.
"""

from __future__ import annotations

import argparse
import ast
import re
import zipfile
from pathlib import Path


MALAYSIA_15_ADDITIONS = [
    "bumps-warning",
    "bus-stop",
    "camera-operation-zone",
    "cow-nearby-warning",
    "height-limit",
    "no-parking",
    "parking-area",
    "towing-area",
    "chevron-left",
    "chevron-right",
    "crossroad-left-warning",
    "crossroad-right-warning",
    "road-narrows-left-warning",
    "road-narrows-right-warning",
    "roadway-diverges-warning",
]
ORIGINAL_49_OVERLAPS = [
    "bicycle-path",
    "uturn-lane",
    "no-left",
    "no-right",
    "no-overtaking",
    "no-uturn",
    "no-horn",
    "traffic-light-ahead",
    "stop-sign",
    "no-entry",
    "give-way",
    "pass-obstacle-on-either-side",
    "pedestrian-crossing-warning",
    "children-crossing-warning",
    "construction-ahead-warning",
    "slippery-road-warning",
    "gated-railway-crossing-ahead-warning",
]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def ordered_names(value: list[str] | dict) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value[index] if index in value else value[str(index)] for index in range(len(value))]
    raise ValueError("data.yaml does not contain a usable names list.")


def names_from_yaml_text(text: str) -> list[str]:
    """Read the simple numeric-name mapping emitted by Roboflow/this project."""
    mapping: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        if line.lstrip().startswith("names:"):
            inline_value = line.split(":", 1)[1].strip()
            if inline_value:
                parsed = ast.literal_eval(inline_value)
                if not isinstance(parsed, list) or not all(isinstance(name, str) for name in parsed):
                    raise ValueError("Inline names value is not a string list.")
                return parsed
            in_names = True
            continue
        if in_names:
            match = re.match(r"^\s*(\d+):\s*(.+?)\s*$", line)
            if match:
                mapping[int(match.group(1))] = match.group(2).strip().strip("'\"")
                continue
            if line and not line.startswith((" ", "\t")):
                break
    if not mapping:
        raise ValueError("data.yaml does not contain a numeric names mapping.")
    return [mapping[index] for index in range(len(mapping))]


def read_source_names(source_zip: zipfile.ZipFile) -> list[str]:
    yaml_members = [name for name in source_zip.namelist() if name == "data.yaml"]
    if not yaml_members:
        raise ValueError("Source ZIP does not contain data.yaml.")
    return names_from_yaml_text(source_zip.read(yaml_members[0]).decode("utf-8"))


def read_target_names(path: Path) -> list[str]:
    return names_from_yaml_text(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument(
        "--target-yaml",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "data.yaml",
        help="Canonical 63-class MYSignVoice config.",
    )
    parser.add_argument(
        "--include-original-overlaps",
        action="store_true",
        help="Also retain the 17 source classes that match the original 49 MYSignVoice classes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    approved_classes = MALAYSIA_15_ADDITIONS.copy()
    if args.include_original_overlaps:
        approved_classes.extend(ORIGINAL_49_OVERLAPS)
    target_names = read_target_names(args.target_yaml)
    target_ids = {name: index for index, name in enumerate(target_names)}
    missing_target = [name for name in approved_classes if name not in target_ids]
    if missing_target:
        raise ValueError(f"Approved names missing from target config: {missing_target}")

    with zipfile.ZipFile(args.source_zip) as source:
        source_names = read_source_names(source)
        source_ids = {name: index for index, name in enumerate(source_names)}
        missing_source = [name for name in approved_classes if name not in source_ids]
        if missing_source:
            raise ValueError(
                "The source version does not contain every approved renamed class: "
                + ", ".join(missing_source)
            )

        image_members = set(source.namelist())
        accepted: list[tuple[str, bytes, str, bytes]] = []
        rejected_mixed = 0
        for label_member in source.namelist():
            parts = Path(label_member).parts
            if len(parts) != 3 or parts[0] not in {"train", "valid", "test"} or parts[1] != "labels":
                continue
            raw_lines = source.read(label_member).decode("utf-8").splitlines()
            if not raw_lines:
                continue

            remapped_lines = []
            mixed = False
            for raw_line in raw_lines:
                values = raw_line.split()
                if len(values) != 5:
                    mixed = True
                    break
                source_id = int(values[0])
                if source_id < 0 or source_id >= len(source_names):
                    mixed = True
                    break
                source_name = source_names[source_id]
                if source_name not in approved_classes:
                    mixed = True
                    break
                remapped_lines.append(" ".join([str(target_ids[source_name]), *values[1:]]))
            if mixed:
                rejected_mixed += 1
                continue

            stem = Path(label_member).stem
            image_prefix = f"{parts[0]}/images/{stem}"
            image_member = next(
                (
                    name
                    for name in image_members
                    if name.startswith(image_prefix)
                    and Path(name).suffix.lower() in IMAGE_SUFFIXES
                ),
                None,
            )
            if image_member is None:
                raise FileNotFoundError(f"No image found for {label_member}")
            accepted.append(
                (
                    image_member,
                    source.read(image_member),
                    label_member,
                    ("\n".join(remapped_lines) + "\n").encode("utf-8"),
                )
            )

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    target_yaml = "\n".join(
        [
            "path: .",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            f"nc: {len(target_names)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(target_names)],
            "",
        ]
    )
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("data.yaml", target_yaml)
        for image_member, image_data, label_member, label_data in accepted:
            output.writestr(image_member, image_data)
            output.writestr(label_member, label_data)
        output.writestr(
            "README_FILTER.txt",
            "This archive contains only the approved Malaysian MYSignVoice classes.\n"
            "Images with any source class outside that approved list were excluded.\n",
        )

    per_class = {name: 0 for name in approved_classes}
    for _, _, _, label_data in accepted:
        for line in label_data.decode("utf-8").splitlines():
            per_class[target_names[int(line.split()[0])]] += 1
    print(f"Created {args.output_zip}")
    print(f"Retained {len(accepted)} labelled images; excluded {rejected_mixed} mixed-label images.")
    for name in approved_classes:
        print(f"{name}: {per_class[name]} boxes")


if __name__ == "__main__":
    main()
