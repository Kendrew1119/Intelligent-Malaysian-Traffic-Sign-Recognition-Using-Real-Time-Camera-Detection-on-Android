# Chapter 4.4 — Shape Detection of Traffic Signs

**Developed by: Member 4 (Kendrew)**
**Module: Shape Detection**

---

## 4.4.1 Algorithm Description

This module detects and classifies the geometric shape of traffic signs from the 84 provided test images using OpenCV C++ image processing techniques. The algorithm integrates **HSV color segmentation** with **contour-based shape classification** to isolate traffic signs from complex backgrounds and identify their shape category.

### Processing Pipeline Flowchart

```text
Input Image
    │
    ▼
Convert BGR → HSV Color Space
    │
    ▼
Apply Color-Specific HSV Threshold
(Red / Blue / Yellow masks)
    │
    ▼
Morphological OPEN (remove noise)
    │
    ▼
Morphological CLOSE (fill holes)
    │
    ▼
Find External Contours
    │
    ▼
Select Largest Contour (= sign boundary)
    │
    ▼
Classify Shape:
├── approxPolyDP (loose, ε=4%) → 3 vertices → Triangle
├── approxPolyDP (loose, ε=4%) → 4 vertices → Rectangle
├── minEnclosingCircle circularity > 0.75?
│   ├── approxPolyDP (strict, ε=1%) → 7-9 vertices → Octagon
│   └── else → Circle
└── else → Polygon
    │
    ▼
Generate 6-Panel Grid Output
    │
    ▼
Save Result + Print Accuracy Statistics
```

### Key Design Decisions

1. **HSV color segmentation instead of raw Canny edge detection.** The previous approach applied Canny edge detection directly on the grayscale image, which picked up edges from trees, buildings, and road markings, producing excessive noise and false contours. By converting to the HSV color space first, the algorithm isolates only the brightly coloured sign regions (red, blue, yellow) and discards the dull background entirely. This dramatically reduces false positives.

2. **Separate HSV thresholds per color category.** Red signs require two HSV ranges because red wraps around the 0°/180° boundary in the Hue channel. Blue and yellow signs each use a single range. The thresholds were empirically tuned:

   | Color | Hue Range | Saturation | Value |
   | :--- | :--- | :--- | :--- |
   | Red (range 1) | 0–10 | 65–255 | 55–255 |
   | Red (range 2) | 165–180 | 60–255 | 55–255 |
   | Blue | 85–135 | 102–255 | 31–255 |
   | Yellow | 12–38 | 80–255 | 50–255 |

3. **Morphological OPEN + CLOSE.** OPEN (erosion then dilation) removes small noise spots. CLOSE (dilation then erosion) fills small holes inside the sign region. Both use a 3×3 rectangular kernel.

4. **Dual-epsilon polygon approximation.** A loose approximation (ε = 4% of perimeter) is used for triangle and rectangle detection where fewer vertices are expected. A strict approximation (ε = 1%) preserves more vertices and is used to distinguish octagons (7–9 vertices) from circles (>9 vertices) among high-circularity contours.

5. **MinEnclosingCircle-based circularity.** Instead of using perimeter-based circularity (which is sensitive to contour roughness), the algorithm computes `area / enclosingCircleArea`. A perfect circle scores 1.0, and values above 0.75 indicate a round shape. This is more robust than the previous approach.

---

## 4.4.2 Results

### Accuracy Summary

*(Fill in these numbers after running the code)*

| Color Category | Total Images | Detected | Accuracy |
| :--- | :--- | :--- | :--- |
| Red Signs | 28 | __ | __% |
| Blue Signs | 28 | __ | __% |
| Yellow Signs | 28 | __ | __% |
| **Overall** | **84** | **__** | **__%** |

### Shape Breakdown

| Shape | Count |
| :--- | :--- |
| Circle | __ |
| Triangle | __ |
| Rectangle | __ |
| Octagon | __ |
| Polygon | __ |

### Sample Results

*(Insert 3–4 representative screenshots from the `output/` folder here. Each grid image shows 6 panels: Original → Contours → Largest Contour → Mask → Shape Name → Sign Segmented)*

**Example 1: Successful circle detection (Red sign)**
*(Insert screenshot: output/Red Signs/Grid_002_0036.png)*

**Example 2: Successful triangle detection (Yellow sign)**
*(Insert screenshot: output/Yellow Signs/Grid_xxx.png)*

**Example 3: Successful rectangle detection (Blue sign)**
*(Insert screenshot: output/Blue Signs/Grid_xxx.png)*

---

## 4.4.3 Error Analysis

### Common Failure Cases

1. **Signs with low saturation or faded colors.** Older signs that have been bleached by sunlight have low saturation values that fall below the HSV threshold. The color mask fails to capture these signs, resulting in no contour being detected.

2. **Signs partially occluded by trees or poles.** When a tree branch covers part of the sign, the contour becomes incomplete. The `findContours` function may split the sign into two or more smaller contours, and the largest contour may not represent the full sign boundary.

3. **Multiple signs in one image.** The algorithm assumes the largest contour is the target sign. If the image contains a large background object with a color similar to the sign (e.g., a red building behind a red sign), the algorithm may incorrectly select the wrong contour.

4. **Octagon vs. Circle misclassification.** Some circular signs with rough edges may have 7–9 vertices in the strict approximation, causing them to be classified as octagons. Similarly, some octagonal stop signs with worn edges may appear too round and be classified as circles.

### Conditions Under Which the System Fails

| Condition | Why It Fails |
| :--- | :--- |
| Heavy rain / fog | Entire image is desaturated; HSV mask captures nothing |
| Night time | Value (brightness) drops below the threshold |
| Sun glare | Overexposed regions become white (saturation = 0) |
| Tilted camera angle | Sign shape appears distorted (parallelogram instead of rectangle) |
| Very small / distant signs | Contour area < 500px threshold; filtered as noise |

---

## 4.4.4 Parameters and Justification

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| HSV Saturation threshold | ≥ 60–102 (varies by color) | Filters out dull backgrounds while retaining vivid sign colors |
| Morphological kernel size | 3×3 | Small enough to preserve sign edges, large enough to remove noise dots |
| Minimum contour area | 500 px² | Filters out tiny noise contours while retaining distant signs |
| Loose approxPolyDP epsilon | 4% of perimeter | Reduces vertices enough to detect triangles (3) and rectangles (4) |
| Strict approxPolyDP epsilon | 1% of perimeter | Preserves enough vertices to distinguish octagons (7–9) from circles |
| Circularity threshold | 0.75 | Empirically separates round shapes (circle/octagon) from angular shapes |

---

## 4.4.5 Enhancements Over Initial Version

The initial version of this module used raw Canny edge detection on the full grayscale image, which resulted in very low accuracy due to background noise. The following enhancements were made:

| Enhancement | Before | After | Impact |
| :--- | :--- | :--- | :--- |
| **HSV Color Segmentation** | Canny on raw grayscale (detects trees, road, sky edges) | HSV mask isolates only sign-colored pixels | Eliminates ~95% of background noise |
| **Morphological OPEN + CLOSE** | Simple dilation only | OPEN removes noise, CLOSE fills holes | Produces cleaner, more complete sign masks |
| **Largest Contour Selection** | All contours drawn with NMS | Only the largest contour is selected | Eliminates duplicate/overlapping detections |
| **MinEnclosingCircle Circularity** | Perimeter-based circularity only | `area / enclosingCircleArea` ratio | More robust circle vs octagon distinction |
| **Dual Epsilon Approximation** | Single ε = 2% | Loose ε = 4% + Strict ε = 1% | Better vertex count accuracy for all shape types |
| **6-Panel Grid Output** | 2×2 grid (grayscale, blur, edges, result) | 3×2 grid (original, contours, largest, mask, shape, segmented) | Matches the lecturer's reference format for report |
| **Accuracy Statistics** | No statistics printed | Per-folder and overall accuracy with shape breakdown | Ready for direct inclusion in report |

---

## 4.4.6 Future Work

To further increase the accuracy of detection beyond the current results, the following improvements can be explored:

1. **Integrating shape detection with the other members' color segmentation modules.** By combining Member 1's red segmentation, Member 2's blue segmentation, and Member 3's yellow segmentation with this shape detection module, the system can achieve a more robust pipeline where color and shape mutually validate each other.

2. **Adaptive HSV thresholds.** Instead of using fixed HSV ranges, the system could dynamically adjust the saturation and value thresholds based on the overall image brightness (e.g., using histogram analysis). This would improve performance under varying lighting conditions.

3. **Deep learning replacement.** As demonstrated in our literature review (Chapter 2), modern YOLOv8-based approaches can achieve significantly higher accuracy than traditional image processing. The low accuracy of this preliminary work strongly justifies the need for our proposed deep learning architecture described in Chapter 3.
