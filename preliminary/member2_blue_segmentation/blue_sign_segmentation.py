# ============================================
# [Member 2] blue_sign_segmentation.py
# ============================================
# Detects blue Malaysian traffic signs with OpenCV HSV colour segmentation.
#
# Usage:
#   python blue_sign_segmentation.py [image-or-folder]
#
# The argument is optional. If omitted, the program searches a few likely
# relative locations for a "Color Inputs/Blue Signs" folder, so it can run
# with no argument as long as the working directory is somewhere inside the repo.
#
# Examples:
#   python blue_sign_segmentation.py
#   python blue_sign_segmentation.py "../../Color Inputs/Blue Signs"
#   python blue_sign_segmentation.py "../../Color Inputs/Blue Signs/example.jpg"
#
# Displays a 6-panel grid (Original, Blue Mask, All Contours, Largest
# Contour, Filled Mask, Segmented Sign) live in a window for each image.
# Nothing is written to disk. Press any key to advance to the next image,
# or ESC to quit early.
#
# A segmentation counts as SUCCESS only when the extracted region looks like a
# COMPLETE sign: it must sit inside the frame, have no rival contour, and fill
# its own best-fit shape (ellipse for circular signs, rectangle for rectangular
# ones). Partial captures - a crescent or half-disc left behind by glare - are
# reported as FAILED with the reason printed alongside.
# ============================================

import cv2
import numpy as np
import os
import sys

# --- Constants (matching the C++ code) ---
MINIMUM_CONTOUR_AREA = 500.0
MINIMUM_ASPECT_RATIO = 0.45
MAXIMUM_ASPECT_RATIO = 2.25
PANEL_SIZE           = 300

# --- Success (completeness) thresholds ---
# Loose gate used only to keep a contour in the candidate list.
CANDIDATE_SOLIDITY   = 0.50
# Strict gates applied when deciding SUCCESS vs FAILED.
# Solidity: how much of its own convex hull the raw contour occupies.
# A whole disc scores ~0.95; a crescent or C-shape left by glare scores far lower.
SUCCESS_SOLIDITY     = 0.80
# Ellipse fill: hull area / best-fit ellipse area. ~1.0 for a full disc,
# stays ~1.0 for a disc seen at an angle. A half-disc drops well below.
MINIMUM_ELLIPSE_FILL = 0.85
# Rect fill: hull area / min-area-rectangle area. ~1.0 for rectangular
# signs, 0.785 for any circle/ellipse/half-disc - so it only ever rescues
# genuinely rectangular signs.
MINIMUM_RECT_FILL    = 0.90


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
    """CLAHE + GaussianBlur + HSV thresholding with morphological cleanup."""
    # CLAHE preprocessing to normalize lighting (handles overcast/shaded signs)
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # GaussianBlur to reduce high-frequency noise before HSV conversion
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Blue HSV range matching Member 4's tested values
    # H=[85,135]: captures full range of Malaysian blue signs
    # S=[80,255]: balanced - catches most signs without too much sky noise
    # V=[40,255]: catches shaded/darker signs
    lower = np.array([85,  80,  40], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological OPEN and CLOSE using a 5x5 elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
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
    success  = False

    img   = cv2.resize(source, (PANEL_SIZE, PANEL_SIZE))
    black = np.zeros_like(img)

    # 1. Get blue mask
    mask = get_blue_mask(img)

    # 2. Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Filter contours by area, size, aspect ratio, and candidate solidity
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= MINIMUM_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 25 or h <= 25:
            continue
        aspect_ratio = float(w) / h
        if aspect_ratio < MINIMUM_ASPECT_RATIO or aspect_ratio > MAXIMUM_ASPECT_RATIO:
            continue

        # Solidity check: reject noisy/irregular contours.
        # A real sign is solid, not fragmented background noise.
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = (area / hull_area) if hull_area > 0 else 0
        if solidity > CANDIDATE_SOLIDITY:
            valid_contours.append(cnt)

    # 4. Initialise 6 panels
    p1 = img.copy()                              # Original
    p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Blue Mask
    p3 = black.copy()                            # All Contours
    p4 = black.copy()                            # Largest Contour (hull outline)
    p5 = black.copy()                            # Filled Mask
    p6 = black.copy()                            # Segmented Sign

    # Draw all valid contours in magenta on p3
    cv2.drawContours(p3, valid_contours, -1, (255, 0, 255), 2)

    reason = []

    # 5. Process the largest valid contour
    if len(valid_contours) > 0:
        detected = True

        largest_idx = 0
        max_area    = 0.0
        for i, cnt in enumerate(valid_contours):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area    = area
                largest_idx = i

        # Use convex hull on the largest contour for cleaner segmentation
        # (from shape_detection.cpp - fixes fragmented contours from white symbols)
        hull     = cv2.convexHull(valid_contours[largest_idx])
        hull_vec = [hull]

        cv2.drawContours(p4, hull_vec, -1, (255, 255, 255), 2)
        cv2.drawContours(p5, hull_vec, -1, (255, 255, 255), cv2.FILLED)

        filled_gray = cv2.cvtColor(p5, cv2.COLOR_BGR2GRAY)
        p6 = cv2.bitwise_and(img, img, mask=filled_gray)

        # --- Gate 1: frame-edge and size checks ---
        bx, by, bw, bh = cv2.boundingRect(valid_contours[largest_idx])
        touches_edge = (bx <= 1 or by <= 1 or
                        (bx + bw) >= PANEL_SIZE - 1 or
                        (by + bh) >= PANEL_SIZE - 1)
        too_large = max_area > (PANEL_SIZE * PANEL_SIZE * 0.85)

        has_competitor = False
        for i, cnt in enumerate(valid_contours):
            if i == largest_idx:
                continue
            if cv2.contourArea(cnt) >= 0.60 * max_area:
                has_competitor = True
                break

        # --- Gate 2: completeness checks ---
        # These are what separate "the whole sign was extracted" from
        # "a chunk of the sign was extracted". Without them a crescent left
        # behind by glare passes as SUCCESS.
        hull_area = cv2.contourArea(hull)
        solidity  = (max_area / hull_area) if hull_area > 0.0 else 0.0

        # Best-fit ellipse: stays near 1.0 for discs viewed head-on OR at an
        # angle, so it tolerates perspective while still rejecting half-discs.
        ellipse_fill = 0.0
        if len(hull) >= 5:
            fitted       = cv2.fitEllipse(hull)
            ellipse_area = np.pi * 0.25 * fitted[1][0] * fitted[1][1]
            if ellipse_area > 0.0:
                ellipse_fill = hull_area / ellipse_area

        # Min-area rectangle: only a genuinely rectangular sign approaches 1.0.
        min_rect      = cv2.minAreaRect(hull)
        min_rect_area = min_rect[1][0] * min_rect[1][1]
        rect_fill     = (hull_area / min_rect_area) if min_rect_area > 0.0 else 0.0

        is_solid    = solidity     >= SUCCESS_SOLIDITY
        whole_shape = (ellipse_fill >= MINIMUM_ELLIPSE_FILL or
                       rect_fill    >= MINIMUM_RECT_FILL)

        if touches_edge:    reason.append("touches frame edge")
        if too_large:       reason.append("region too large")
        if has_competitor:  reason.append("competing contour")
        if not is_solid:    reason.append("fragmented contour")
        if not whole_shape: reason.append("partial shape")

        success = (not touches_edge and not too_large and
                   not has_competitor and is_solid and whole_shape)

        if not success:
            print(f"    metrics: solidity={solidity:.2f}"
                  f" ellipseFill={ellipse_fill:.2f}"
                  f" rectFill={rect_fill:.2f}")

    # 6. Add standard labels
    add_label(p1, "Original")
    add_label(p2, "Blue Mask")
    add_label(p3, "All Contours")
    add_label(p4, "Largest Contour")
    add_label(p5, "Filled Mask")
    add_label(p6, "Segmented Sign")

    # 7. Stack panels into 3x2 grid
    top_row    = np.hstack([p1, p2, p3])
    bottom_row = np.hstack([p4, p5, p6])
    grid       = np.vstack([top_row, bottom_row])

    status = "SUCCESS" if success else "FAILED"
    line   = (f"[{filename}] "
              f"{'Blue Sign Detected' if detected else 'No Blue Sign Detected'} - {status}")
    if not success and reason:
        line += f" ({', '.join(reason)})"
    print(line)

    return grid, detected, success


def main():
    print("=" * 40)
    print(" Member 2: Blue Sign Segmentation (Python)")
    print(" MYSignVoice Preliminary Work")
    print("=" * 40)

    # Find the input path: use explicit argument or auto-discover
    input_path = ""
    for arg in sys.argv[1:]:
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

    total_images     = 0
    total_detected   = 0
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
        if detected: total_detected   += 1
        if success:  total_successful += 1

        cv2.imshow(window_name, grid)
        key = cv2.waitKey(0)
        if key == 27:  # ESC quits early
            print("Stopped early by user.")
            break

    cv2.destroyAllWindows()

    # Print summary
    detection_rate = (100.0 * total_detected   / total_images) if total_images > 0 else 0.0
    success_rate   = (100.0 * total_successful / total_images) if total_images > 0 else 0.0

    print(f"\n{'=' * 40}")
    print(" BLUE SIGN SEGMENTATION SUMMARY")
    print(f"{'=' * 40}")
    print(f" Total images processed:         {total_images}")
    print(f" Total blue signs detected:      {total_detected}")
    print(f" Total successful segmentations: {total_successful}")
    print(f" Detection Rate: {detection_rate:.1f}%")
    print(f" Success Rate:   {success_rate:.1f}%")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
