"""Create a safe labelled MYSignVoice import from ROAD SIGN CLASSIFICATION v1.

Only verified source-to-target label matches that are currently below the
100-image collection threshold are retained. Images containing another source
class are excluded so unrecognised signs cannot become background examples.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from filter_malaysia_15class_export import IMAGE_SUFFIXES, read_source_names, read_target_names


SOURCE_TO_TARGET = {
    "Go Straight Or Right": "straight-or-right",
    "No Horn": "no-horn",
    "No Overtaking": "no-overtaking",
    "Railway Crossing": "railway-crossing-ahead-warning",
    "Roundabout Ahead": "roundabout",
    "Sharp Right Turn": "sharp-right-turn-warning",
    "Slow": "slowdown-warning",
    "Speed Limit -30 km-h-": "speed-limit-30",
    "Speed Limit -40 km-h-": "speed-limit-40",
    "Speed Limit -50 km-h-": "speed-limit-50",
    "Speed Limit -60 km-h-": "speed-limit-60",
    "Speed Limit -80 km-h-": "speed-limit-80",
    "Steep Descent": "steep-descent-warning",
    "Work Ahead": "construction-ahead-warning",
    "Zigzag Road Ahead": "winding-road-warning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument(
        "--target-yaml",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "data.yaml",
        help="Canonical MYSignVoice class configuration used to validate target names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_names = read_target_names(args.target_yaml)
    missing_targets = sorted(set(SOURCE_TO_TARGET.values()) - set(target_names))
    if missing_targets:
        raise ValueError(f"Missing target classes: {', '.join(missing_targets)}")

    target_subset = [name for name in target_names if name in SOURCE_TO_TARGET.values()]
    target_ids = {name: index for index, name in enumerate(target_subset)}

    with zipfile.ZipFile(args.source_zip) as source:
        source_names = read_source_names(source)
        source_ids = {name: index for index, name in enumerate(source_names)}
        missing_sources = sorted(set(SOURCE_TO_TARGET) - set(source_ids))
        if missing_sources:
            raise ValueError(f"Missing source classes: {', '.join(missing_sources)}")

        members = set(source.namelist())
        accepted: list[tuple[str, bytes, str, bytes]] = []
        excluded = 0
        for label_member in source.namelist():
            parts = Path(label_member).parts
            if len(parts) != 3 or parts[0] not in {"train", "valid", "test"} or parts[1] != "labels":
                continue

            raw_lines = source.read(label_member).decode("utf-8").splitlines()
            if not raw_lines:
                continue
            remapped_lines: list[str] = []
            safe = True
            for raw_line in raw_lines:
                values = raw_line.split()
                if len(values) != 5:
                    safe = False
                    break
                source_id = int(values[0])
                if source_id < 0 or source_id >= len(source_names):
                    safe = False
                    break
                target_name = SOURCE_TO_TARGET.get(source_names[source_id])
                if target_name is None:
                    safe = False
                    break
                remapped_lines.append(" ".join([str(target_ids[target_name]), *values[1:]]))
            if not safe:
                excluded += 1
                continue

            stem = Path(label_member).stem
            prefix = f"{parts[0]}/images/{stem}"
            image_member = next(
                (
                    name
                    for name in members
                    if name.startswith(prefix) and Path(name).suffix.lower() in IMAGE_SUFFIXES
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
    data_yaml = "\n".join(
        [
            "path: .",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            f"nc: {len(target_subset)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(target_subset)],
            "",
        ]
    )
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("data.yaml", data_yaml)
        for image_member, image_data, label_member, label_data in accepted:
            output.writestr(image_member, image_data)
            output.writestr(label_member, label_data)

    counts = {name: 0 for name in target_subset}
    for _, _, _, label_data in accepted:
        for line in label_data.decode("utf-8").splitlines():
            counts[target_subset[int(line.split()[0])]] += 1
    print(f"Created {args.output_zip}")
    print(f"Retained {len(accepted)} labelled images; excluded {excluded} mixed-label images.")
    for name in target_subset:
        print(f"{name}: {counts[name]} boxes")


if __name__ == "__main__":
    main()
