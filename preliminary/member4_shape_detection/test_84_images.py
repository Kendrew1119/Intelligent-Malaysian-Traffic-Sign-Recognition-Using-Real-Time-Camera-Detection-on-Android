# ============================================
# [Member 4] test_84_images.py — HYBRID VERSION
# ============================================
# Purpose: Combine OpenCV HSV color detection
#          with YOLOv8 deep learning to test
#          all 84 provided images.
#
# Pipeline:
#   1. OpenCV HSV masking finds colored regions
#   2. Regions are cropped and enlarged
#   3. YOLO runs on BOTH the full image AND
#      the cropped regions for maximum accuracy
#   4. Results are merged and de-duplicated
#   5. A 6-panel grid is saved for each image
#
# Usage:
#   pip install ultralytics opencv-python numpy
#   python test_84_images.py
# ============================================

from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import time

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "best.pt"
TEST_DIR = "../../Color Inputs"
OUTPUT_DIR = "yolo_results"
GRID_DIR = "hybrid_grids"         # 6-panel grids for report
CONF_THRESHOLD = 0.15             # Lower threshold to catch more signs
CROP_CONF_THRESHOLD = 0.10        # Even lower for cropped regions
MIN_CONTOUR_AREA = 300            # Minimum HSV region area in pixels
CROP_PADDING = 30                 # Extra pixels around detected region

# ============================================
# HSV COLOR RANGES for Malaysian signs
# ============================================
# Red has TWO ranges in HSV (wraps around 0/180)
RED_LOWER_1 = np.array([0, 70, 50])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 70, 50])
RED_UPPER_2 = np.array([180, 255, 255])

# Blue signs
BLUE_LOWER = np.array([90, 50, 50])
BLUE_UPPER = np.array([130, 255, 255])

# Yellow/Orange warning signs
YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([45, 255, 255])


def detect_colored_regions(img, color_hint="auto"):
    """
    Use OpenCV HSV masking to find candidate sign regions.
    Returns list of bounding boxes [(x, y, w, h), ...]
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    masks = []

    if color_hint in ["auto", "red"]:
        mask_r1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        mask_r2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        masks.append(mask_r1 | mask_r2)

    if color_hint in ["auto", "blue"]:
        masks.append(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))

    if color_hint in ["auto", "yellow"]:
        masks.append(cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER))

    # Combine all color masks
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m

    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Add padding
        x = max(0, x - CROP_PADDING)
        y = max(0, y - CROP_PADDING)
        bw = min(w - x, bw + 2 * CROP_PADDING)
        bh = min(h - y, bh + 2 * CROP_PADDING)

        # Filter out very thin or very wide regions (not likely signs)
        aspect = bw / max(bh, 1)
        if aspect > 5 or aspect < 0.2:
            continue

        regions.append((x, y, bw, bh))

    return regions, combined


def run_yolo_on_crop(model, img, region, full_img_shape):
    """
    Crop a region from the image, resize it to 640x640,
    run YOLO on it, then map detections back to full image coords.
    """
    x, y, w, h = region
    crop = img[y:y+h, x:x+w]

    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return []

    # Resize crop to 640x640 for better detection of small signs
    crop_resized = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)

    results = model(crop_resized, conf=CROP_CONF_THRESHOLD, verbose=False)
    detections = results[0].boxes

    mapped = []
    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        # Map bounding box back to original image coordinates
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        scale_x = w / 640.0
        scale_y = h / 640.0
        bx1 = int(bx1 * scale_x) + x
        by1 = int(by1 * scale_y) + y
        bx2 = int(bx2 * scale_x) + x
        by2 = int(by2 * scale_y) + y

        mapped.append({
            "cls_id": cls_id,
            "cls_name": cls_name,
            "conf": conf,
            "bbox": (bx1, by1, bx2, by2),
            "source": "crop"
        })

    return mapped


def merge_detections(full_dets, crop_dets, iou_threshold=0.3):
    """
    Merge detections from full-image YOLO and cropped-region YOLO.
    Remove duplicates using IoU (Intersection over Union).
    """
    all_dets = list(full_dets)

    for cd in crop_dets:
        is_duplicate = False
        for i, fd in enumerate(all_dets):
            iou = compute_iou(cd["bbox"], fd["bbox"])
            if iou > iou_threshold:
                is_duplicate = True
                # Keep whichever has higher confidence
                if cd["conf"] > fd["conf"]:
                    all_dets[i] = cd
                break

        if not is_duplicate:
            all_dets.append(cd)

    return all_dets


def compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes (x1,y1,x2,y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)


def draw_detections(img, detections):
    """Draw bounding boxes and labels on image."""
    result = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["cls_name"]
        conf = det["conf"]
        source = det.get("source", "full")

        # Color: Green for full-image YOLO, Cyan for crop-enhanced
        color = (0, 255, 0) if source == "full" else (255, 255, 0)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.0%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return result


def create_grid(original, hsv_mask, regions_img, annotated, cls_name, conf, filename):
    """
    Create a 6-panel grid image for the report:
    [ Original      | HSV Mask        | HSV Regions     ]
    [ YOLO Result   | Classification  | Combined Output ]
    """
    cell_w, cell_h = 320, 240

    # Resize all panels
    p1 = cv2.resize(original, (cell_w, cell_h))
    p2 = cv2.resize(cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR), (cell_w, cell_h))
    p3 = cv2.resize(regions_img, (cell_w, cell_h))
    p4 = cv2.resize(annotated, (cell_w, cell_h))

    # Panel 5: Classification text
    p5 = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    if cls_name:
        cv2.putText(p5, "Detected:", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Word-wrap long class names
        words = cls_name.split()
        y_pos = 80
        line = ""
        for word in words:
            test_line = line + " " + word if line else word
            if len(test_line) > 20:
                cv2.putText(p5, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 30
                line = word
            else:
                line = test_line
        if line:
            cv2.putText(p5, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(p5, f"Confidence: {conf:.1%}", (10, y_pos + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
    else:
        cv2.putText(p5, "NOT DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(p5, "Sign not in", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(p5, "training dataset", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Panel 6: Pipeline info
    p6 = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cv2.putText(p6, "Hybrid Pipeline:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
    cv2.putText(p6, "1. HSV Color Mask", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(p6, "2. Region Extraction", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(p6, "3. YOLO Full Image", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(p6, "4. YOLO on Crops", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(p6, "5. Merge & Deduplicate", (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(p6, f"File: {filename}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # Add labels to each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    panels = [
        (p1, "Original"),
        (p2, "HSV Color Mask"),
        (p3, "Detected Regions"),
        (p4, "YOLO + HSV Result"),
        (p5, "Classification"),
        (p6, "Pipeline Info"),
    ]
    for panel, title in panels:
        cv2.rectangle(panel, (0, 0), (cell_w - 1, cell_h - 1), (80, 80, 80), 1)
        cv2.rectangle(panel, (0, 0), (len(title) * 10 + 10, 22), (40, 40, 40), -1)
        cv2.putText(panel, title, (5, 16), font, 0.45, (200, 200, 200), 1)

    # Assemble grid
    row1 = np.hstack([panels[0][0], panels[1][0], panels[2][0]])
    row2 = np.hstack([panels[3][0], panels[4][0], panels[5][0]])
    grid = np.vstack([row1, row2])

    return grid


def main():
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found!")
        print("Please download best.pt from Google Drive after training.")
        sys.exit(1)

    # Check test directory exists
    if not os.path.exists(TEST_DIR):
        print(f"ERROR: Test directory '{TEST_DIR}' not found!")
        sys.exit(1)

    # Load YOLO model
    print("=" * 60)
    print(" MYSignVoice — HYBRID OpenCV + YOLOv8 Test on 84 Images")
    print("=" * 60)
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRID_DIR, exist_ok=True)

    # Map subfolder names to color hints for HSV
    folder_color_map = {
        "Red Signs": "red",
        "Blue Signs": "blue",
        "Yellow Signs": "yellow"
    }

    total_images = 0
    total_detected = 0
    total_yolo_only = 0       # Detected by full-image YOLO alone
    total_crop_enhanced = 0   # Additional detections from crop
    class_counts = {}
    results_log = []

    start_time = time.time()

    for subfolder, color_hint in folder_color_map.items():
        folder_path = os.path.join(TEST_DIR, subfolder)
        if not os.path.exists(folder_path):
            print(f"  WARNING: Folder not found: {folder_path}")
            continue

        out_subfolder = os.path.join(OUTPUT_DIR, subfolder)
        grid_subfolder = os.path.join(GRID_DIR, subfolder)
        os.makedirs(out_subfolder, exist_ok=True)
        os.makedirs(grid_subfolder, exist_ok=True)

        folder_total = 0
        folder_detected = 0

        print(f"\nProcessing: {subfolder} (HSV hint: {color_hint})")
        print("-" * 60)

        for filename in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg", ".bmp"]:
                continue

            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue

            total_images += 1
            folder_total += 1

            # ========== STEP 1: OpenCV HSV Color Detection ==========
            regions, hsv_mask = detect_colored_regions(img, color_hint)

            # Draw regions on a copy for visualization
            regions_vis = img.copy()
            for (rx, ry, rw, rh) in regions:
                cv2.rectangle(regions_vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                cv2.putText(regions_vis, "HSV Region", (rx, ry - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # ========== STEP 2: Full-Image YOLO Detection ==========
            full_results = model(img, conf=CONF_THRESHOLD, verbose=False)
            full_dets = []
            for box in full_results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                full_dets.append({
                    "cls_id": cls_id,
                    "cls_name": model.names[cls_id],
                    "conf": float(box.conf[0]),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "source": "full"
                })

            # ========== STEP 3: Crop-Enhanced YOLO Detection ==========
            crop_dets = []
            for region in regions:
                crop_results = run_yolo_on_crop(model, img, region, img.shape)
                crop_dets.extend(crop_results)

            # ========== STEP 4: Merge All Detections ==========
            merged = merge_detections(full_dets, crop_dets)

            # Track detection sources
            if len(merged) > 0:
                total_detected += 1
                folder_detected += 1

                has_full = any(d["source"] == "full" for d in merged)
                has_crop = any(d["source"] == "crop" for d in merged)

                if has_full:
                    total_yolo_only += 1
                if has_crop and not has_full:
                    total_crop_enhanced += 1

                # Count each class
                best_det = max(merged, key=lambda d: d["conf"])
                best_cls = best_det["cls_name"]
                best_conf = best_det["conf"]
                class_counts[best_cls] = class_counts.get(best_cls, 0) + 1

                results_log.append({
                    "folder": subfolder,
                    "file": filename,
                    "class": best_cls,
                    "confidence": best_conf,
                    "source": best_det["source"],
                    "num_detections": len(merged)
                })
            else:
                best_cls = None
                best_conf = 0
                results_log.append({
                    "folder": subfolder,
                    "file": filename,
                    "class": "NOT_DETECTED",
                    "confidence": 0,
                    "source": "none",
                    "num_detections": 0
                })

            # ========== STEP 5: Draw and Save Results ==========
            annotated = draw_detections(img, merged)

            # Save annotated image
            save_path = os.path.join(out_subfolder, filename)
            cv2.imwrite(save_path, annotated)

            # Save 6-panel grid for report
            grid = create_grid(img, hsv_mask, regions_vis, annotated,
                               best_cls, best_conf, filename)
            grid_path = os.path.join(grid_subfolder, f"Grid_{filename}")
            cv2.imwrite(grid_path, grid)

            # Console output
            if merged:
                det_info = f"{len(merged)} sign(s): {best_cls} ({best_conf:.0%})"
                source_info = f" [via {best_det['source']}]"
            else:
                det_info = "NO DETECTION"
                source_info = ""
            print(f"  [{filename}] {det_info}{source_info}")

        # Per-folder summary
        folder_acc = (100.0 * folder_detected / folder_total) if folder_total > 0 else 0
        print(f"  >> {subfolder}: {folder_detected}/{folder_total} detected ({folder_acc:.1f}%)")

    # ============================================
    # OVERALL RESULTS SUMMARY
    # ============================================
    elapsed = time.time() - start_time
    overall_acc = (100.0 * total_detected / total_images) if total_images > 0 else 0

    print("\n" + "=" * 60)
    print(" HYBRID DETECTION RESULTS SUMMARY")
    print("=" * 60)
    print(f" Total images tested:      {total_images}")
    print(f" Images with detection:    {total_detected}")
    print(f" Overall detection rate:   {overall_acc:.1f}%")
    print(f" ---- Detection Source Breakdown ----")
    print(f"   Full-image YOLO only:   {total_yolo_only}")
    print(f"   Crop-enhanced (bonus):  {total_crop_enhanced}")
    print(f" Time taken:               {elapsed:.1f}s")
    print("-" * 60)
    print(" Class breakdown:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"   {cls_name}: {count}")
    print("=" * 60)
    print(f" Annotated images:  {OUTPUT_DIR}/")
    print(f" 6-Panel grids:     {GRID_DIR}/")
    print("=" * 60)

    # Save report to text file
    report_path = os.path.join(OUTPUT_DIR, "accuracy_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MYSignVoice — Hybrid OpenCV + YOLOv8 Detection Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Full-image confidence threshold: {CONF_THRESHOLD}\n")
        f.write(f"Crop-enhanced confidence threshold: {CROP_CONF_THRESHOLD}\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Detected: {total_detected}\n")
        f.write(f"Detection rate: {overall_acc:.1f}%\n")
        f.write(f"  YOLO-only detections: {total_yolo_only}\n")
        f.write(f"  Crop-enhanced bonus:  {total_crop_enhanced}\n\n")
        f.write("Class breakdown:\n")
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {cls_name}: {count}\n")
        f.write("\n\nDetailed results:\n")
        f.write("-" * 60 + "\n")
        for entry in results_log:
            f.write(f"  [{entry['folder']}] {entry['file']} -> "
                    f"{entry['class']} ({entry['confidence']:.2f}) "
                    f"[source: {entry['source']}]\n")

    print(f"\n Report saved to: {report_path}")


if __name__ == "__main__":
    main()
