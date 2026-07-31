# ============================================
# [Member 3] yellow_sign_segmentation.py
# ============================================
# Module: Yellow Sign Segmentation Using Color Information
# Owner: Member 3
#
# Purpose:
#   - Python replica of the C++ yellow_sign_segmentation.cpp code.
#   - Bypasses Windows Application Control policy blocking custom .exe files.
#   - Generates the exact same 6-panel grid output images for Report Chapter 4.
#
# Usage:
#   python yellow_sign_segmentation.py
# ============================================

import cv2
import numpy as np
import os
import sys

def get_yellow_mask(src):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    
    # Yellow range in HSV (stricter):
    lower = np.array([18, 100, 80], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological OPEN and CLOSE using a 5x5 elliptical struct element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def process_image(src, filename, out_path, show_steps=False):
    # Resize to 300x300 for consistent layout
    img = cv2.resize(src, (300, 300))
    black = np.zeros_like(img)
    
    # 1. Get binary mask
    yellow_mask = get_yellow_mask(img)
    
    # 2. Find contours
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Contour Filtering (Geometry Check)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 25 and h > 25:
                aspect_ratio = float(w) / h
                if 0.4 <= aspect_ratio <= 1.6:
                    valid_contours.append(cnt)
                    
    # Initialize 6 panels
    p1 = img.copy()
    p2 = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR) # Yellow Mask
    p3 = black.copy()                                 # All Contours
    p4 = black.copy()                                 # Largest Contour
    p5 = black.copy()                                 # Filled Mask
    p6 = black.copy()                                 # Segmented Sign (black background)
    
    # Draw valid filtered contours in magenta on p3
    cv2.drawContours(p3, valid_contours, -1, (255, 0, 255), 2)
    
    detected = False
    success = False
    
    # 4. Find largest yellow contour from valid contours
    if len(valid_contours) > 0:
        detected = True
        largest_idx = 0
        max_area = 0
        for i, cnt in enumerate(valid_contours):
            a = cv2.contourArea(cnt)
            if a > max_area:
                max_area = a
                largest_idx = i
                
        if max_area > 500:
            # Draw largest outline on p4 in white
            cv2.drawContours(p4, valid_contours, largest_idx, (255, 255, 255), 2)
            
            # Create filled mask on p5 in white
            cv2.drawContours(p5, valid_contours, largest_idx, (255, 255, 255), cv2.FILLED)
            
            # Segment sign (bitwise AND using p5's filled mask)
            p6 = cv2.bitwise_and(img, img, mask=p5[:, :, 0])
            
            # 5. Automatic Segmentation Status Heuristics
            # check 1: touches or is within 3 pixels of any image edge
            touches_edge = False
            lx, ly, lw, lh = cv2.boundingRect(valid_contours[largest_idx])
            if lx <= 3 or ly <= 3 or (lx + lw) >= 297 or (ly + lh) >= 297:
                touches_edge = True
                
            # check 2: occupies more than 45% of total image area
            too_large = max_area > (90000 * 0.45)
            
            # check 3: another valid contour exists whose area is at least 30% of largest
            has_competitor = False
            for i, cnt in enumerate(valid_contours):
                if i == largest_idx:
                    continue
                a = cv2.contourArea(cnt)
                if a >= 0.30 * max_area:
                    has_competitor = True
                    break
                    
            if not touches_edge and not too_large and not has_competitor:
                success = True
                
    # 6. Draw segmentation status label on panel 6
    status_color = (0, 255, 0) if success else (0, 0, 255)
    status_text = "Segmentation: SUCCESS" if success else "Segmentation: FAILED"
    cv2.putText(p6, status_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
            
    # 7. Add standard labels
    def add_label(panel, text):
        h, w = panel.shape[:2]
        cv2.putText(panel, text, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    add_label(p1, "Original")
    add_label(p2, "Yellow Mask")
    add_label(p3, "All Contours")
    add_label(p4, "Largest Contour")
    add_label(p5, "Filled Mask")
    add_label(p6, "Segmented Sign")
    
    # 8. Stack panels into 3x2 grid:
    #    Top row: Original, Yellow Mask, All Contours
    #    Bottom row: Largest Contour, Filled Mask, Segmented Sign
    top_row = np.hstack([p1, p2, p3])
    bottom_row = np.hstack([p4, p5, p6])
    grid = np.vstack([top_row, bottom_row])
    
    # Save the result
    cv2.imwrite(out_path, grid)
    
    # Optional interactive visualization
    if show_steps:
        win_name = "Yellow Sign Segmentation - " + filename
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 900, 600)
        cv2.imshow(win_name, grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    if detected:
        print(f"  [{filename}] Yellow Sign Detected - {'SUCCESS' if success else 'FAILED'}")
    else:
        print(f"  [{filename}] No Yellow Sign Detected - FAILED")
        
    return detected, success

def main():
    print("=" * 40)
    print(" Member 3: Yellow Sign Segmentation (Python)")
    print(" MYSignVoice Preliminary Work")
    print("=" * 40)
    
    # Check command-line arguments
    show_steps = len(sys.argv) > 1 and sys.argv[1] == "--show"
    if show_steps:
        print("Mode: Step-by-step visualization enabled")
        
    # Find base folder dynamically
    base_dir = ""
    candidates = [
        "../../Color Inputs/Yellow Signs",
        "../Color Inputs/Yellow Signs",
        "Color Inputs/Yellow Signs"
    ]
    for cand in candidates:
        if os.path.exists(cand):
            base_dir = cand
            break
            
    if not base_dir:
        print("ERROR: Could not locate directory 'Color Inputs/Yellow Signs'!")
        sys.exit(-1)
        
    print(f"Reading images from: {base_dir}")
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    total_images = 0
    total_detected = 0
    total_successful = 0
    total_failed = 0
    
    for filename in sorted(os.listdir(base_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".bmp"]:
            continue
            
        filepath = os.path.join(base_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"  WARNING: Failed to read image {filename}")
            continue
            
        total_images += 1
        out_path = os.path.join(output_dir, f"Grid_{filename}")
        
        det, succ = process_image(img, filename, out_path, show_steps)
        if det:
            total_detected += 1
        if succ:
            total_successful += 1
        else:
            total_failed += 1
                
    detection_rate = (100.0 * total_detected / total_images) if total_images > 0 else 0
    success_rate = (100.0 * total_successful / total_images) if total_images > 0 else 0
    
    print("\n" + "=" * 40)
    print(" YELLOW SIGN SEGMENTATION SUMMARY")
    print("=" * 40)
    print(f" Total images processed: {total_images}")
    print(f" Total yellow signs detected: {total_detected}")
    print(f" Total successful segmentations: {total_successful}")
    print(f" Total failed segmentations: {total_failed}")
    print(f" Detection Rate:         {detection_rate:.1f}%")
    print(f" Success Rate:           {success_rate:.1f}%")
    print(f" Output folder location: {output_dir}/")
    print("=" * 40)

if __name__ == "__main__":
    main()
