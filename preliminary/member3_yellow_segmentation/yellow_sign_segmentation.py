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
    
    # Yellow range in HSV: H: [12, 38], S: [80, 255], V: [50, 255]
    lower = np.array([12, 80, 50], dtype=np.uint8)
    upper = np.array([38, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological OPEN (3x3 rect kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Morphological CLOSE (3x3 rect kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def classify_shape(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    
    # Loose approximation for triangle (3) and rectangle (4)
    approx_loose = cv2.approxPolyDP(contour, 0.04 * peri, True)
    # Strict approximation for octagon (7-9) and circle (>9)
    approx_strict = cv2.approxPolyDP(contour, 0.01 * peri, True)
    
    vertices_loose = len(approx_loose)
    vertices_strict = len(approx_strict)
    
    # Circularity using minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(contour)
    enclosing_area = np.pi * radius * radius
    circularity = area / enclosing_area if enclosing_area > 0 else 0
    
    if vertices_loose == 3:
        return "Triangle"
    elif vertices_loose == 4:
        return "Rectangle"
    elif circularity > 0.75:
        if 7 <= vertices_strict <= 9:
            return "Octagon"
        else:
            return "Circle"
    else:
        return "Polygon"

def process_image(src, filename, out_path):
    # Resize to 300x300 for consistent layout
    img = cv2.resize(src, (300, 300))
    black = np.zeros_like(img)
    
    # 1. Get binary mask
    yellow_mask = get_yellow_mask(img)
    
    # 2. Find contours
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize 6 panels
    p1 = img.copy()
    p2 = black.copy()
    p3 = black.copy()
    p4 = black.copy()
    p5 = black.copy()
    p6 = black.copy()
    
    shape_name = "No Shape"
    
    # Draw all contours in magenta on p2
    cv2.drawContours(p2, contours, -1, (255, 0, 255), 2)
    
    # 3. Find largest yellow contour
    if len(contours) > 0:
        largest_idx = 0
        max_area = 0
        for i, cnt in enumerate(contours):
            a = cv2.contourArea(cnt)
            if a > max_area:
                max_area = a
                largest_idx = i
                
        if max_area > 500:
            # Draw largest outline on p3
            cv2.drawContours(p3, contours, largest_idx, (255, 255, 255), 2)
            
            # Create filled mask on p4
            cv2.drawContours(p4, contours, largest_idx, (255, 255, 255), cv2.FILLED)
            
            # 4. Classify shape
            shape_name = classify_shape(contours[largest_idx])
            
            # Draw filled green shape on p5
            cv2.drawContours(p5, contours, largest_idx, (0, 255, 0), cv2.FILLED)
            
            # 5. Segment sign (bitwise AND)
            p6 = cv2.bitwise_and(img, img, mask=p4[:, :, 0])
            
    # 6. Add labels
    def add_label(panel, text):
        h, w = panel.shape[:2]
        cv2.putText(panel, text, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    add_label(p1, "Original")
    add_label(p2, "All Contours")
    add_label(p3, "Largest Contour")
    add_label(p4, "Yellow Mask")
    add_label(p5, f"Shape: {shape_name}")
    add_label(p6, "Sign Segmented")
    
    # 7. Stack panels into 3x2 grid
    top_row = np.hstack([p1, p2, p3])
    bottom_row = np.hstack([p4, p5, p6])
    grid = np.vstack([top_row, bottom_row])
    
    # Save the result
    cv2.imwrite(out_path, grid)
    print(f"  [{filename}] Detected: {shape_name}")
    return shape_name

def main():
    print("=" * 40)
    print(" Member 3: Yellow Sign Segmentation (Python)")
    print(" MYSignVoice Preliminary Work")
    print("=" * 40)
    
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
    shapes = {"Circle": 0, "Triangle": 0, "Rectangle": 0, "Octagon": 0, "Polygon": 0}
    
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
        
        shape = process_image(img, filename, out_path)
        
        if shape != "No Shape":
            total_detected += 1
            if shape in shapes:
                shapes[shape] += 1
                
    detection_rate = (100.0 * total_detected / total_images) if total_images > 0 else 0
    print("\n" + "=" * 40)
    print(" YELLOW SIGN SEGMENTATION SUMMARY")
    print("=" * 40)
    print(f" Total images processed: {total_images}")
    print(f" Total shapes detected:  {total_detected}")
    print(f" Detection Rate:         {detection_rate:.1f}%")
    print("-" * 40)
    print(" Detected Shape Breakdown:")
    for s_name, count in shapes.items():
        print(f"   {s_name}:    {count}")
    print("=" * 40)
    print(f" Results saved to directory: {output_dir}/")
    print("=" * 40)

if __name__ == "__main__":
    main()
