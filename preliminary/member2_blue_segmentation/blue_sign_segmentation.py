# ============================================
# [Member 2] blue_sign_segmentation.py
# ============================================
# Module: Blue Sign Segmentation Using Color Information
# Owner: Member 2
#
# Purpose:
#   - Python replica of the C++ blue_sign_segmentation.cpp code.
#   - Bypasses Windows Application Control policy blocking custom .exe files.
#   - Generates the exact same 6-panel grid output images for Report Chapter 4.
#
# Usage:
#   python blue_sign_segmentation.py                 (auto-finds Blue Signs folder)
#   python blue_sign_segmentation.py --show          (with step-by-step visualization)
#   python blue_sign_segmentation.py "path/to/images"
# ============================================

import cv2
import numpy as np
import os
import sys

# --- Constants (matching the C++ code) ---
MINIMUM_CONTOUR_AREA = 500.0
MINIMUM_ASPECT_RATIO = 0.45
MAXIMUM_ASPECT_RATIO = 2.25
PANEL_SIZE = 300


def find_default_input_path():
    """Tries a list of likely folder locations so the program can run with no argument."""
    candidates = [
        "Color Inputs/Blue Signs",
        "Color Inputs/Traffic signs/Blue Signs",
        "../Color Inputs/Blue Signs",
        "../Color Inputs/Traffic signs/Blue Signs",
        "../../Color Inputs/Blue Signs",
        "../../Color Inputs/Traffic signs/Blue Signs",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def get_blue_mask(src):
    """HSV thresholding for blue pixels with morphological cleanup."""
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # Blue HSV range: H=[100,130], S=[80,255], V=[50,255]
    lower = np.array([100, 80, 50], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological OPEN and CLOSE using a 5x5 elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def add_label(panel, text):
    """Adds a white label at the bottom-left of a panel."""
    h = panel.shape[0]
    cv2.putText(panel, text, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)


def build_grid(source, filename):
    """
    Builds the 6-panel grid for one image.
    Returns (grid, detected, success).
    """
    detected = False
    success = False

    img = cv2.resize(source, (PANEL_SIZE, PANEL_SIZE))
    black = np.zeros_like(img)

    # 1. Get blue mask
    mask = get_blue_mask(img)

    # 2. Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Filter contours by area, size, and aspect ratio
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= MINIMUM_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 25 or h <= 25:
            continue
        aspect_ratio = float(w) / h
        if MINIMUM_ASPECT_RATIO <= aspect_ratio <= MAXIMUM_ASPECT_RATIO:
            valid_contours.append(cnt)

    # Initialize 6 panels
    p1 = img.copy()                                      # Original
    p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)           # Blue Mask
    p3 = black.copy()                                    # All Contours
    p4 = black.copy()                                    # Largest Contour
    p5 = black.copy()                                    # Filled Mask
    p6 = black.copy()                                    # Segmented Sign

    # Draw all valid contours in magenta on p3
    cv2.drawContours(p3, valid_contours, -1, (255, 0, 255), 2)

    # 4. Find the largest valid blue contour
    if len(valid_contours) > 0:
        detected = True

        largest_idx = 0
        max_area = 0.0
        for i, cnt in enumerate(valid_contours):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                largest_idx = i

        # Draw largest contour outline on p4 in white
        cv2.drawContours(p4, valid_contours, largest_idx, (255, 255, 255), 2)

        # Create filled mask on p5 in white
        cv2.drawContours(p5, valid_contours, largest_idx, (255, 255, 255), cv2.FILLED)

        # Segment sign using the filled mask
        filled_gray = cv2.cvtColor(p5, cv2.COLOR_BGR2GRAY)
        p6 = cv2.bitwise_and(img, img, mask=filled_gray)

        # 5. Segmentation quality checks
        bx, by, bw, bh = cv2.boundingRect(valid_contours[largest_idx])
        touches_edge = (bx <= 3 or by <= 3 or
                        (bx + bw) >= PANEL_SIZE - 3 or
                        (by + bh) >= PANEL_SIZE - 3)
        too_large = max_area > (PANEL_SIZE * PANEL_SIZE * 0.45)

        has_competitor = False
        for i, cnt in enumerate(valid_contours):
            if i == largest_idx:
                continue
            if cv2.contourArea(cnt) >= 0.30 * max_area:
                has_competitor = True
                break

        success = not touches_edge and not too_large and not has_competitor

    # 6. Draw segmentation status label on panel 6
    status_color = (0, 255, 0) if success else (0, 0, 255)
    status_text = "Segmentation: SUCCESS" if success else "Segmentation: FAILED"
    cv2.putText(p6, status_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, status_color, 1, cv2.LINE_AA)

    # 7. Add standard labels
    add_label(p1, "Original")
    add_label(p2, "Blue Mask")
    add_label(p3, "All Contours")
    add_label(p4, "Largest Contour")
    add_label(p5, "Filled Mask")
    add_label(p6, "Segmented Sign")

    # 8. Stack panels into 3x2 grid
    top_row = np.hstack([p1, p2, p3])
    bottom_row = np.hstack([p4, p5, p6])
    grid = np.vstack([top_row, bottom_row])

    label = "Blue Sign Detected" if detected else "No Blue Sign Detected"
    result = "SUCCESS" if success else "FAILED"
    print(f"  [{filename}] {label} - {result}")

    return grid, detected, success


def main():
    print("=" * 40)
    print(" Member 2: Blue Sign Segmentation (Python)")
    print(" MYSignVoice Preliminary Work")
    print("=" * 40)

    # Parse arguments
    show_steps = "--show" in sys.argv
    if show_steps:
        print("Mode: Step-by-step visualization enabled")

    # Find the input path: use explicit argument or auto-discover
    input_path = ""
    for arg in sys.argv[1:]:
        if arg != "--show":
            input_path = arg
            break

    if not input_path:
        input_path = find_default_input_path()
        if not input_path:
            print("ERROR: Could not auto-locate 'Color Inputs/Blue Signs'!")
            print("Tried multiple relative paths. You can pass the path manually:")
            print(f"  python {sys.argv[0]} \"path/to/Blue Signs\"")
            sys.exit(-1)
        print(f"No argument given, using auto-detected folder: {input_path}")

    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        sys.exit(1)

    print(f"Reading images from: {input_path}")

    # Collect image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = []
    if os.path.isfile(input_path):
        if os.path.splitext(input_path)[1].lower() in image_extensions:
            images.append(input_path)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in image_extensions:
                    images.append(os.path.join(root, f))
    images.sort()

    if not images:
        print(f"No supported image files found in: {input_path}")
        sys.exit(1)

    # Also save grids to an output folder (for the report)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    total_images = 0
    total_detected = 0
    total_successful = 0

    window_name = "Blue Sign Segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)

    for image_path in images:
        source = cv2.imread(image_path)
        if source is None:
            print(f"  WARNING: Could not read image: {image_path}")
            continue

        total_images += 1
        filename = os.path.basename(image_path)

        grid, detected, success = build_grid(source, filename)
        if detected:
            total_detected += 1
        if success:
            total_successful += 1

        # Save grid to output folder (for report screenshots)
        out_path = os.path.join(output_dir, f"Grid_{filename}")
        cv2.imwrite(out_path, grid)

        # Always show live display
        cv2.imshow(window_name, grid)
        key = cv2.waitKey(0)
        if key == 27:  # ESC quits early
            print("Stopped early by user.")
            break

    cv2.destroyAllWindows()

    # Print summary
    detection_rate = (100.0 * total_detected / total_images) if total_images > 0 else 0.0
    success_rate = (100.0 * total_successful / total_images) if total_images > 0 else 0.0

    print(f"\n{'=' * 40}")
    print(" BLUE SIGN SEGMENTATION SUMMARY")
    print(f"{'=' * 40}")
    print(f" Total images processed:        {total_images}")
    print(f" Total blue signs detected:     {total_detected}")
    print(f" Total successful segmentations: {total_successful}")
    print(f" Detection Rate: {detection_rate:.1f}%")
    print(f" Success Rate:   {success_rate:.1f}%")
    print(f" Output folder location: {output_dir}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
