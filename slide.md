# Member 4 Presentation Slides — Chapter 3 and Chapter 4

**Suggested duration:** 4–6 minutes
**Recommended slides:** 8 slides

---

## Slide 1 — Member 4: Image Processing and YOLO26s Plan

**MYSignVoice: Malaysian Traffic Sign Recognition for Visually Impaired Pedestrians**

- Member 4 role: preliminary OpenCV image processing, shape detection, YOLO26s training and evaluation.
- Goal: detect Malaysian traffic signs from a camera and provide accessible feedback.
- This presentation covers:
  - Chapter 3: proposed system approach
  - Chapter 4: preliminary OpenCV shape-detection result

**Say:** “My preliminary work proves that colour segmentation and shape analysis can locate sign candidates. The final system will add YOLO26s to recognise the exact sign class.”

---

## Slide 2 — Chapter 3: Proposed System Overview

```text
Camera Frame
    → Frame Preparation
    → OpenCV Image Processing
    → YOLO26s Detection and 49-Class Recognition
    → Confidence / Stability Filter
    → Text-to-Speech and Vibration Alert
```

- OpenCV provides explainable colour and shape information.
- YOLO26s provides the final class label, confidence score and bounding box.
- The output is designed for visually impaired pedestrians: spoken sign meaning and warning vibration.

**Image to insert:** Screenshot or redraw of the **overall block diagram** from `enhanced_block_diagram.md` section 3.2.1.

**Say:** “OpenCV and YOLO have different roles. OpenCV identifies sign-like regions; YOLO is the final recogniser because many signs share the same colour and shape.”

---

## Slide 3 — Why Image Processing Alone Is Not Enough

| Classical OpenCV preliminary method | Final YOLO26s model |
|---|---|
| Detects red, blue and yellow regions | Recognises the exact traffic-sign class |
| Identifies Circle, Triangle, Rectangle or Octagon | Distinguishes 63 classes, such as different speed limits |
| Fast and explainable | More robust to different pictograms and complex scenes |
| Can be affected by glare, shadows and similar-colour backgrounds | Learns visual features from labelled training data |

- Example: speed limit 30 and speed limit 60 are both red circles.
- Shape detection cannot identify the number inside the sign.
- Therefore, the YOLO26s CNN is required for final recognition.

**Image to insert:** One red circular speed-limit image, for example `Color Inputs/Red Signs/005_0030.png`.

---

## Slide 4 — Chapter 3: YOLO26s Dataset and Training Plan

```text
Collect 63-class images
    → Annotate bounding boxes
    → Split train / validation / test
    → Training-only augmentation
    → Fine-tune YOLO26s at 640 in Google Colab
    → Evaluate and export to NCNN for Android
```

- Target: 49 Malaysian traffic-sign classes.
- Use YOLO-format bounding-box annotations.
- Augment only training images: brightness, rotation, scale, blur, noise and partial occlusion.
- Evaluate using precision, recall, mAP@0.5 and confusion matrix.
- Deploy the selected model on Android using NCNN.

**Important point:** The 84 existing images are for preliminary OpenCV testing only. They are not enough to train a reliable 63-class YOLO model. Collect at least 50 original labelled images per class before augmentation.

**Image to insert:** Screenshot/redraw of the **YOLO training and deployment diagram** from `enhanced_block_diagram.md` section 3.2.4.

---

## Slide 5 — Chapter 4: Preliminary OpenCV Method

```text
Input image
    → Convert BGR to HSV
    → Red / Blue / Yellow colour mask
    → Morphological open + close
    → Find contours
    → Aspect ratio and solidity filtering
    → Geometric shape classification
```

- HSV segmentation isolates sign colours more effectively than using grayscale edges alone.
- Morphological opening removes small noise; closing fills small holes.
- The largest valid contour is analysed using polygon vertices, circle-fill ratio and bounding-box extent.
- Output shapes: Circle, Triangle, Rectangle, Octagon or Polygon.

**Image to insert:** Screenshot/redraw of the **OpenCV image-processing diagram** from `enhanced_block_diagram.md` section 3.2.3.

---

## Slide 6 — Chapter 4: Static Image Demonstration (`--show`)

- Run `shape_detection.cpp --show`.
- The program displays a six-panel grid for each static image:
  1. Original image
  2. Grayscale image
  3. Canny edges — visual explanation only
  4. HSV colour mask — actual contour source
  5. Classified shape
  6. Segmented sign

**Image to insert:** Use one successful saved grid image, preferably a clear example such as:

`preliminary/member4_shape_detection/output/Red Signs/Grid_002_0036.png`

**Say:** “Although Canny edges are shown, the classification uses the HSV colour mask. This reduces background edge noise.”

---

## Slide 7 — Chapter 4: Live Camera Demonstration

- Build and run the separate `camera_detection.cpp` program.
- It processes red, blue and yellow masks in real time.
- Camera-specific steps:
  - limited gray-world white-balance correction;
  - contour area, aspect-ratio and solidity filters;
  - a three-frame stability check to reduce flickering.
- Press **M** to show: Original, HSV Mask, Contours and Final Output.
- Press **Q** or **Esc** to exit.

**Image to insert:** Take a screenshot during your live demo with **M** enabled.

**Say:** “This is a colour-and-shape demonstration, not the final 63-class recognition model.”

---

## Slide 8 — Preliminary Results and Limitations

| Category | Correct / Total | Accuracy |
|---|---:|---:|
| Red Signs | 25 / 28 | 89.3% |
| Blue Signs | 26 / 28 | 92.9% |
| Yellow Signs | 27 / 28 | 96.4% |
| **Overall** | **78 / 84** | **92.9%** |

- Six images were classified with the wrong geometric shape.
- Main causes: pixelation, glare, incomplete HSV masks and backgrounds with similar colours.
- Example errors: blue sky/background merging with blue signs; vegetation joining yellow sign masks.
- These limitations justify the final YOLO26s recognition stage.

**Image to insert:** One failure grid, preferably:

`preliminary/member4_shape_detection/output/Blue Signs/Grid_023_1_0005.png`

Add a small caption: **“Glare and incomplete blue mask cause Polygon output.”**

---

## Slide 9 — Conclusion and Next Step

- Preliminary OpenCV pipeline successfully demonstrates segmentation and shape detection: **92.9% (78/84)** geometric shape accuracy.
- The method is useful for candidate detection and visual explanation.
- It cannot identify the exact sign pictogram/class by itself.
- Next step: collect balanced 63-class data, annotate, augment the training split and train YOLO26s at a 640-pixel baseline.
- Final system: server-side YOLO26s/OpenVINO recognition + laptop-browser camera + web display/speech output.

**Closing sentence:** “The preliminary image-processing result provides the foundation, while the validated YOLO26s server pipeline will recognise the actual Malaysian traffic-sign class.”

---

## Presentation Checklist

- [ ] Insert the overall block diagram on Slide 2.
- [ ] Insert the YOLO training diagram on Slide 4.
- [ ] Insert the OpenCV flow diagram on Slide 5.
- [ ] Insert one successful six-panel `--show` grid on Slide 6.
- [ ] Capture one live-camera screenshot with the **M** mask view for Slide 7.
- [ ] Insert the blue failure grid on Slide 8.
- [ ] Prepare one red, one blue and one yellow printed/on-screen sign for the live camera demonstration.
- [ ] Say “shape-classification accuracy” for the 92.9% result; do not describe it as 63-class recognition accuracy.
