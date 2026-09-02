"""Compare V2 and V3 predictions on the 84 supplied sign images.

The leading three-digit source code identifies the expected sign class for this
fixed lecturer-provided set. This is a top-class recognition check, not mAP:
the images do not have reviewed YOLO bounding-box ground truth in this folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


EXPECTED_BY_CODE = {
    "000": "speed-limit-5",
    "001": "speed-limit-15",
    "002": "speed-limit-30",
    "003": "speed-limit-40",
    "004": "speed-limit-50",
    "005": "speed-limit-60",
    "007": "speed-limit-80",
    "008": "no-straight-or-left",
    "010": "no-straight",
    "011": "no-left",
    "012": "no-left-and-right",
    "013": "no-right",
    "014": "no-overtaking",
    "015": "no-uturn",
    "016": "no-cars",
    "017": "no-horn",
    "020": "straight-or-right",
    "021": "straight-only",
    "022": "left-turn-only",
    "023": "left-or-right",
    "024": "right-turn-only",
    "026": "pass-right",
    "027": "roundabout",
    "028": "cars-only",
    "029": "use-horn",
    "030": "bicycle-path",
    "031": "uturn-lane",
    "032": "pass-obstacle-on-either-side",
    "033": "traffic-light-ahead",
    "034": "general-warning",
    "035": "pedestrian-crossing-warning",
    "036": "bicycle-warning",
    "037": "children-crossing-warning",
    "038": "sharp-right-turn-warning",
    "040": "steep-descent-warning",
    "042": "slowdown-warning",
    "043": "crossroad-right-warning",
    "045": "village-ahead-warning",
    "046": "reverse-turn-warning",
    "047": "railway-crossing-ahead-warning",
    "048": "construction-ahead-warning",
    "049": "winding-road-warning",
    "050": "gated-railway-crossing-ahead-warning",
    "051": "accident-prone-area-warning",
    "052": "stop-sign",
    "055": "no-entry",
    "056": "give-way",
    "057": "stop-for-inspection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True, help="V2 detections.csv")
    parser.add_argument("--v3", type=Path, required=True, help="V3 detections.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def image_key(source: str) -> str:
    return Path(source).name


def expected_class(filename: str) -> str:
    match = re.match(r"^(\d{3})", filename)
    if not match or match.group(1) not in EXPECTED_BY_CODE:
        raise ValueError(f"No reviewed class mapping for source image: {filename}")
    return EXPECTED_BY_CODE[match.group(1)]


def load_predictions(csv_path: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = image_key(row["source"])
            grouped.setdefault(key, [])
            if row["class_name"]:
                grouped[key].append(
                    {
                        "class_name": row["class_name"],
                        "confidence": float(row["confidence"]),
                    }
                )
    return dict(grouped)


def model_fields(
    predictions: list[dict[str, object]], expected: str, prefix: str
) -> dict[str, object]:
    ordered = sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True)
    top = ordered[0] if ordered else None
    expected_items = [item for item in ordered if item["class_name"] == expected]
    wrong_items = [item for item in ordered if item["class_name"] != expected]
    return {
        f"{prefix}_top_class": top["class_name"] if top else "",
        f"{prefix}_top_confidence": round(float(top["confidence"]), 6) if top else "",
        f"{prefix}_top_correct": bool(top and top["class_name"] == expected),
        f"{prefix}_expected_found": bool(expected_items),
        f"{prefix}_expected_confidence": (
            round(float(expected_items[0]["confidence"]), 6) if expected_items else ""
        ),
        f"{prefix}_detection_count": len(ordered),
        f"{prefix}_wrong_detection_count": len(wrong_items),
    }


def summarize(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    total = len(rows)
    top_correct = sum(bool(row[f"{prefix}_top_correct"]) for row in rows)
    expected_found = sum(bool(row[f"{prefix}_expected_found"]) for row in rows)
    misses = sum(int(row[f"{prefix}_detection_count"]) == 0 for row in rows)
    wrong_only = sum(
        int(row[f"{prefix}_detection_count"]) > 0
        and not bool(row[f"{prefix}_expected_found"])
        for row in rows
    )
    wrong_detections = sum(int(row[f"{prefix}_wrong_detection_count"]) for row in rows)
    return {
        "images": total,
        "top_class_correct": top_correct,
        "top_class_accuracy": top_correct / total if total else 0.0,
        "images_where_expected_class_was_found": expected_found,
        "expected_class_found_rate": expected_found / total if total else 0.0,
        "images_with_no_detection": misses,
        "images_with_only_wrong_classes": wrong_only,
        "wrong_class_detections": wrong_detections,
    }


def main() -> None:
    args = parse_args()
    v2 = load_predictions(args.v2)
    v3 = load_predictions(args.v3)
    keys = sorted(set(v2) | set(v3))
    if len(keys) != 84:
        raise SystemExit(f"Expected 84 source images, found {len(keys)}")

    rows: list[dict[str, object]] = []
    for key in keys:
        expected = expected_class(key)
        row: dict[str, object] = {"source_image": key, "expected_class": expected}
        row.update(model_fields(v2.get(key, []), expected, "v2"))
        row.update(model_fields(v3.get(key, []), expected, "v3"))
        row["selection_change"] = (
            row["v2_top_class"] != row["v3_top_class"]
            or row["v2_top_correct"] != row["v3_top_correct"]
        )
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output / "per_image_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "evaluation_scope": (
            "Top predicted class versus the reviewed filename-code mapping for the "
            "84 supplied single-sign images; bounding-box mAP is not calculated."
        ),
        "v2": summarize(rows, "v2"),
        "v3": summarize(rows, "v3"),
        "images_with_changed_top_selection": sum(bool(row["selection_change"]) for row in rows),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Per-image comparison: {comparison_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
