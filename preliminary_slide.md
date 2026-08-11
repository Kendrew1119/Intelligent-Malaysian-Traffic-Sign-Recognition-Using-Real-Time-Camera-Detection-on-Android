# Member 4 Preliminary Presentation Slides — Chapter 3

**Suggested duration:** 3–5 minutes
**Focus:** Completed preliminary colour and shape detection; YOLO26s is the approved next phase.

---

## Slide 1 — Member 4 Preliminary Work

**MYSignVoice: Preliminary Traffic-Sign Colour and Shape Detection**

- Member 4: OpenCV image processing and shape detection.
- Current scope: detect sign colour and broad geometric shape.
- Input: static traffic-sign images and live webcam frames.
- Next phase: YOLO26s at 640 for exact sign-class recognition.

**Say:** “This preliminary prototype focuses on detecting traffic-sign colour and shape. It is the foundation before adding deep learning in the next phase.”

---

## Slide 2 — Preliminary System Overview

```text
Input Image / Webcam Frame
    → HSV Colour Segmentation
    → Morphological Open and Close
    → Contour Detection and Filtering
    → Shape Classification
    → Colour + Shape Result
```

- Detects red, blue and yellow sign-coloured regions.
- Classifies Circle, Triangle, Rectangle, Octagon or Polygon.
- Shows a six-panel result for static images or a live overlay for webcam frames.

**Image to insert:** Screenshot/redraw of the **Preliminary System Block Diagram** from `preliminary_block_diagram.md`, section 3.2.1.

---

## Slide 3 — Image-Processing Method

- Convert the BGR image to HSV colour space.
- Apply a colour threshold to isolate red, blue or yellow pixels.
- Use **3 × 3 morphological open** to remove small noise dots.
- Use **3 × 3 morphological close** to fill small gaps in the sign region.
- Find contours from the cleaned mask.
- Filter contours using area, aspect ratio and solidity.

**Say:** “The HSV colour mask is the real detection method. Canny edges are displayed only to help explain the image-processing steps.”

**Image to insert:** Screenshot/redraw of the **Image-Processing Detail Block Diagram** from `preliminary_block_diagram.md`, section 3.2.3.

---

## Slide 4 — Shape Classification

- The largest valid contour is converted to a convex hull.
- The program measures polygon vertices, circle-fill ratio and bounding-box extent.
- Shape outputs:
  - 3 vertices → Triangle
  - 4 vertices / extent check → Triangle or Rectangle
  - high circle-fill ratio + 7–9 strict vertices → Octagon
  - high circle-fill ratio otherwise → Circle

**Image to insert:** A clear successful output grid, for example:

`preliminary/member4_shape_detection/output/Red Signs/Grid_002_0036.png`

Point to the **Colour Mask**, **Shape**, and **Sign Segmented** panels.

---

## Slide 5 — Static Image Demonstration

- Run: `shape_detection.cpp --show`
- Each image shows six panels:
  1. Original
  2. Grayscale
  3. Canny Edges
  4. Colour Mask
  5. Classified Shape
  6. Sign Segmented
- The result grid is also saved for report evidence.

**Image to insert:** Your own best `--show` screenshot, or use the same successful grid from Slide 4.

---

## Slide 6 — Live Camera Demonstration

- Build and run the separate `camera_detection.cpp` program.
- Uses red, blue and yellow HSV masks on every camera frame.
- Applies limited white-balance correction and a three-frame stability check.
- Press **M** to show Original, HSV Mask, Contours and Final Output.
- Press **Q** or **Esc** to quit.

**Image to insert:** Take a screenshot of your webcam screen while the **M** split view is enabled.

**Say:** “The live output currently identifies colour and shape, not the exact sign class.”

---

## Slide 7 — Preliminary Results

| Category | Correct / Total | Shape Accuracy |
|---|---:|---:|
| Red Signs | 25 / 28 | 89.3% |
| Blue Signs | 26 / 28 | 92.9% |
| Yellow Signs | 27 / 28 | 96.4% |
| **Overall** | **78 / 84** | **92.9%** |

- Six images had an incorrect shape result.
- Main reasons: pixelation, glare, incomplete colour masks and similar-colour backgrounds.

**Image to insert:** A failure example:

`preliminary/member4_shape_detection/output/Blue Signs/Grid_023_1_0005.png`

Caption: **“Glare and a weak blue mask distort the circular contour.”**

---

## Slide 8 — Challenges and Next Phase

### Current challenges

- Background colours can merge with the sign mask.
- Glare, shadows and pixelation deform contours.
- One colour and shape can represent many different signs.
- The largest-contour method is limited when multiple signs appear.

### Next phase

```text
Collect labelled sign images
    → Annotate bounding boxes
    → Train YOLO26s at 640
    → Recognise exact Malaysian traffic-sign class
```

**Closing sentence:** “The preliminary OpenCV system proves that we can locate and analyse sign candidates. YOLO26s will be added next and validated for exact traffic-sign recognition.”

---

## Checklist Before Presenting

- [ ] Insert the two block diagrams from `preliminary_block_diagram.md`.
- [ ] Insert one successful static grid and one failure grid.
- [ ] Take one camera screenshot with **M** enabled.
- [ ] Prepare a red, blue and yellow sign for the live demo.
- [ ] Describe 92.9% as **preliminary shape-classification accuracy**, not YOLO or final sign-recognition accuracy.
- [ ] State clearly that YOLO26s at 640 is the next phase.
