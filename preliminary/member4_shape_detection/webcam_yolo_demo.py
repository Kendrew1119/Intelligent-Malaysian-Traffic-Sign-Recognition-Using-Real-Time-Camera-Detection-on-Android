# ============================================
# [Member 4] webcam_yolo_demo.py — HYBRID VERSION
# ============================================
# Purpose: Real-time webcam detection combining
#          OpenCV HSV preprocessing with YOLOv8.
#
# Pipeline (runs every frame):
#   1. Capture frame from webcam
#   2. OpenCV HSV masking finds colored regions
#   3. YOLO runs on full frame
#   4. YOLO runs on each cropped HSV region
#   5. Merge results & draw on screen
#
# Controls:
#   Press 'q' to quit
#   Press 's' to save a screenshot
#   Press 'm' to toggle HSV mask overlay
#   Press 'h' to toggle hybrid mode (on/off)
#
# Usage:
#   pip install ultralytics opencv-python numpy
#   python webcam_yolo_demo.py
# ============================================

from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import sys

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "best.pt"
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.35             # Higher for camera = less false positives
CROP_CONF_THRESHOLD = 0.25        # Higher for crops too
MIN_CONTOUR_AREA = 1500           # Ignore small noise in video
CROP_PADDING = 20
SCREENSHOT_DIR = "screenshots"
MAX_CROPS_PER_FRAME = 3           # Fewer crops = faster FPS
HYBRID_EVERY_N_FRAMES = 3         # Only run crop-detection every Nth frame

# HSV Color Ranges
RED_LOWER_1 = np.array([0, 70, 50])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 70, 50])
RED_UPPER_2 = np.array([180, 255, 255])
BLUE_LOWER = np.array([90, 50, 50])
BLUE_UPPER = np.array([130, 255, 255])
YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([45, 255, 255])


def detect_colored_regions(img):
    """Detect red, blue, and yellow regions using HSV masking."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # Red (two ranges)
    mask_r1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask_r2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask_red = mask_r1 | mask_r2

    # Blue
    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    # Yellow
    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    combined = mask_red | mask_blue | mask_yellow

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Add padding
        x = max(0, x - CROP_PADDING)
        y = max(0, y - CROP_PADDING)
        bw = min(w - x, bw + 2 * CROP_PADDING)
        bh = min(h - y, bh + 2 * CROP_PADDING)

        # Filter extreme aspect ratios
        aspect = bw / max(bh, 1)
        if aspect > 4 or aspect < 0.25:
            continue

        regions.append((x, y, bw, bh))

        if len(regions) >= MAX_CROPS_PER_FRAME:
            break

    return regions, combined


def run_yolo_on_crop(model, img, region):
    """Crop, resize to 640x640, run YOLO, map coords back."""
    x, y, w, h = region
    crop = img[y:y+h, x:x+w]

    if crop.shape[0] < 20 or crop.shape[1] < 20:
        return []

    crop_resized = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)
    results = model(crop_resized, conf=CROP_CONF_THRESHOLD, verbose=False)

    mapped = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()

        scale_x = w / 640.0
        scale_y = h / 640.0
        bx1 = int(bx1 * scale_x) + x
        by1 = int(by1 * scale_y) + y
        bx2 = int(bx2 * scale_x) + x
        by2 = int(by2 * scale_y) + y

        mapped.append({
            "cls_id": cls_id,
            "cls_name": model.names[cls_id],
            "conf": conf,
            "bbox": (bx1, by1, bx2, by2),
            "source": "crop"
        })

    return mapped


def compute_iou(box1, box2):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)


def merge_detections(full_dets, crop_dets, iou_threshold=0.3):
    """Merge full-image and crop detections, removing duplicates."""
    all_dets = list(full_dets)

    for cd in crop_dets:
        is_duplicate = False
        for i, fd in enumerate(all_dets):
            iou = compute_iou(cd["bbox"], fd["bbox"])
            if iou > iou_threshold:
                is_duplicate = True
                if cd["conf"] > fd["conf"]:
                    all_dets[i] = cd
                break
        if not is_duplicate:
            all_dets.append(cd)

    return all_dets


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found!")
        sys.exit(1)

    # Load model
    print("=" * 60)
    print(" MYSignVoice — HYBRID Real-Time Webcam Demo")
    print("=" * 60)
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")

    # Open webcam
    print(f"Opening camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    print("\nCamera is running!")
    print("  Press 'q' to quit")
    print("  Press 's' to save screenshot")
    print("  Press 'm' to toggle HSV mask overlay")
    print("  Press 'h' to toggle hybrid mode on/off")
    print("-" * 60)

    frame_count = 0
    fps_start = time.time()
    screenshot_count = 0
    show_mask = False
    hybrid_mode = True
    detection_history = []  # Keep last N detections for display stability
    last_crop_dets = []       # Cache crop detections between frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # ========== STEP 1: Full-Image YOLO ==========
        full_results = model(frame, conf=CONF_THRESHOLD, verbose=False)
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

        merged = list(full_dets)

        if hybrid_mode:
            # ========== STEP 2: HSV Region Detection ==========
            regions, hsv_mask = detect_colored_regions(frame)

            # Draw HSV regions (yellow dashed rectangles)
            for (rx, ry, rw, rh) in regions:
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 1)

            # ========== STEP 3: Crop-Enhanced YOLO (every Nth frame only) ==========
            if frame_count % HYBRID_EVERY_N_FRAMES == 0:
                last_crop_dets = []
                for region in regions:
                    crop_results = run_yolo_on_crop(model, frame, region)
                    last_crop_dets.extend(crop_results)

            # ========== STEP 4: Merge (use cached crop dets on skipped frames) ==========
            merged = merge_detections(full_dets, last_crop_dets)

            # Show mask overlay if toggled on
            if show_mask:
                mask_colored = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
                mask_colored[:, :, 0] = 0  # Remove blue channel
                mask_colored[:, :, 2] = 0  # Remove red channel — green mask
                display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)

        # ========== STEP 5: Draw All Detections ==========
        for det in merged:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["cls_name"]
            conf = det["conf"]
            source = det.get("source", "full")

            # Green box for YOLO-only, Cyan for crop-enhanced
            if source == "crop":
                color = (255, 255, 0)  # Cyan
                thickness = 3
            else:
                color = (0, 255, 0)    # Green
                thickness = 2

            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            # Label with background
            label = f"{cls_name} {conf:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - label_size[1] - 8),
                          (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # ========== HUD Overlay ==========
        elapsed = time.time() - fps_start
        fps = frame_count / max(elapsed, 0.001)
        num_dets = len(merged)

        # Top info bar
        mode_text = "HYBRID (OpenCV+YOLO)" if hybrid_mode else "YOLO ONLY"
        info_text = f"FPS: {fps:.1f} | Detections: {num_dets} | Mode: {mode_text}"
        cv2.rectangle(display, (0, 0), (600, 35), (0, 0, 0), -1)
        cv2.putText(display, info_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Bottom controls bar
        h_frame = display.shape[0]
        controls = "q=Quit | s=Screenshot | m=Mask | h=Toggle Hybrid"
        cv2.rectangle(display, (0, h_frame - 30), (520, h_frame), (0, 0, 0), -1)
        cv2.putText(display, controls, (10, h_frame - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Detection history sidebar (show last 5 detections)
        if merged:
            for det in merged:
                detection_history.append(det["cls_name"])
            detection_history = detection_history[-5:]

        sidebar_x = display.shape[1] - 250
        cv2.rectangle(display, (sidebar_x, 0), (display.shape[1], 35 + len(detection_history) * 25), (0, 0, 0), -1)
        cv2.putText(display, "Recent Detections:", (sidebar_x + 5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        for i, name in enumerate(detection_history):
            y_pos = 45 + i * 25
            cv2.putText(display, f"> {name}", (sidebar_x + 5, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Show the frame
        cv2.imshow("MYSignVoice — Hybrid Real-Time Detection", display)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            screenshot_count += 1
            save_path = os.path.join(SCREENSHOT_DIR, f"hybrid_screenshot_{screenshot_count}.png")
            cv2.imwrite(save_path, display)
            print(f"  Screenshot saved: {save_path}")
        elif key == ord('m'):
            show_mask = not show_mask
            print(f"  HSV mask overlay: {'ON' if show_mask else 'OFF'}")
        elif key == ord('h'):
            hybrid_mode = not hybrid_mode
            mode_name = "HYBRID (OpenCV+YOLO)" if hybrid_mode else "YOLO ONLY"
            print(f"  Mode switched to: {mode_name}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Processed {frame_count} frames.")
    if elapsed > 0:
        print(f"Average FPS: {frame_count / elapsed:.1f}")
    if screenshot_count > 0:
        print(f"Screenshots saved to: {SCREENSHOT_DIR}/")


if __name__ == "__main__":
    main()
