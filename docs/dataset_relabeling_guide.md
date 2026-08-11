# MYSignVoice — 49-Class Roboflow Annotation Plan

## Confirmed scope

The reviewed `class.xlsx` inventory covers **49 classes and 84 sign images**:

| Colour group | Classes | Sign images |
|---|---:|---:|
| Blue | 12 | 28 |
| Red | 21 | 28 |
| Yellow | 16 | 28 |
| **Total** | **49** | **84** |

`dataset/data.yaml` is the canonical class-ID mapping. The labels below must be
copied into Roboflow exactly, in this order. They are lowercase and use hyphens.
This class inventory is model-independent: the active detector changes to YOLO26s,
but the 49 IDs and names do not change. IDs are zero-based, and corrected class ID
33 is `pass-obstacle-on-either-side`.

```text
straight-or-right, straight-only, basement-entrance, left-turn-only, left-or-right,
right-turn-only, pass-right, roundabout, cars-only, use-horn, bicycle-path,
uturn-lane, speed-limit-5, speed-limit-15, speed-limit-30, speed-limit-40,
speed-limit-50, speed-limit-60, speed-limit-80, no-straight-or-left, no-straight,
no-left, no-left-and-right, no-right, no-overtaking, no-uturn, no-cars, no-horn,
traffic-light-ahead, stop-sign, no-entry, give-way, stop-for-inspection,
pass-obstacle-on-either-side, general-warning, pedestrian-crossing-warning, bicycle-warning,
children-crossing-warning, sharp-right-turn-warning, steep-descent-warning,
slowdown-warning, t-intersection-right-warning, village-ahead-warning,
winding-road-warning, railway-crossing-ahead-warning, construction-ahead-warning,
slippery-road-warning, gated-railway-crossing-ahead-warning,
accident-prone-area-warning
```

## Phase 1 — Create the project and seed annotations

1. In Roboflow, create a project named `MYSignVoice-49-Signs` with project type
   **Object Detection**.
2. Create all 49 classes using the list above before any annotation begins. Do not
   use spaces, capitals, underscores, or alternative spellings such as `stop` instead
   of `stop-sign`.
3. Upload the 84 supplied images from `Color Inputs/`. Label each visible sign with
   one tight bounding box. The box should include the full sign face and exclude as
   much sky, pole, road, and background as possible.
4. Use the colour and shape columns in `class.xlsx` only as a checking aid. They
   are not separate YOLO classes.
5. Review every annotation at 100% zoom before marking the batch complete. Fix
   boxes that cut off a sign edge, include a large background margin, or have an
   incorrect class.

## Phase 2 — Collect enough original images

The 84 images are a valid labelled **seed set**, but they are not large enough for
49-class recognition: 29 classes currently have only one example and no class has
more than four. Augmentation creates variations; it does not create new sign
appearances or provide independent test evidence.

Collect original photos of the same Malaysian sign designs, preferably taken by the
team in varied locations, distances, lighting, angles, and backgrounds. Aim for at
least **20 original annotated images per class** (980 total, so approximately 896
additional images beyond the seed set). Record the source of each image and ensure
you are allowed to use it.

For each class, reserve different original photos for each split. Do not split
near-duplicate video frames across train and validation/test, and do not add
augmented copies to validation or test.

| Split | Target share | With 20 originals/class |
|---|---:|---:|
| Train | 70% | 14 images/class |
| Validation | 20% | 4 images/class |
| Test | 10% | 2 images/class |

Until each class has at least three distinct originals, a class-aware train/validation/test
split is impossible. Train a small seed model only as a demonstration, not as a final
accuracy result.

### Laptop-webcam domain set and hard negatives

The final demonstration uses a laptop webcam, so part of the original dataset must
come from the same type of camera rather than only downloaded or high-quality phone
images. For every sign class, capture examples at close, medium and far distances,
from several horizontal and vertical angles, and under bright, dim and backlit
conditions. Include normal webcam defects such as autofocus blur, motion blur,
compression, sensor noise and low contrast. If the demonstration uses printed signs
or signs displayed on a monitor, include different print sizes, paper glare, screen
moire and varied backgrounds without reusing the final test frames.

Also collect a dedicated **hard-negative/no-sign set**. These images have no traffic
sign bounding boxes and therefore use an empty YOLO label file. Include red vehicles,
blue advertising boards, sky, yellow vegetation, coloured clothing, circular logos,
triangular objects and traffic lights. Hard negatives teach the model to reject the
same objects that can fool HSV colour segmentation. Capture whole webcam sessions,
then keep complete sessions in only one split so adjacent frames cannot leak between
training and evaluation.

## Phase 3 — Generate the Roboflow version

1. Create a new dataset version after all original images are annotated and reviewed.
2. Set the 70/20/10 split before augmentation.
3. Preprocess with auto-orient and resize with aspect-preserving letterboxing to
   640 × 640. Do not stretch signs into a square.
4. Apply augmentation to the training split only. Use moderate brightness/exposure,
   contrast and gamma changes; mild saturation/value changes; motion or defocus blur;
   sensor noise; JPEG/webcam compression; downsample-and-upsample simulation; scale,
   translation, slight perspective and rotation up to ±8°; and limited occlusion.
   Keep every augmented sign human-readable and keep transformed bounding boxes
   aligned with the sign.
5. Do **not** horizontally or vertically flip the images: flips change left/right,
   turn and directional sign meanings. Do not apply augmentation to validation or
   test images.
6. Export in **Ultralytics YOLO detection format**. If Roboflow labels the option
   **YOLOv8**, that export is still the compatible text-label format consumed by
   YOLO26. In the exported `data.yaml`, verify that `nc` is 49, corrected class ID
   33 is `pass-obstacle-on-either-side`, and every name/order entry matches
   `dataset/data.yaml` before training.

## Phase 4 — Evaluation and reporting

Keep a separate, unseen test split for final mAP, precision, and recall. The provided
84 images can be used for a clearly-labelled seed/demo test only if they were not
used for training; otherwise report them as a training-set demonstration, not an
independent test score. Also capture a small set of new real-world photographs for
the project’s beyond-the-provided-images evaluation.

For the webcam pipeline, evaluate the same saved videos in three modes: pure
full-frame YOLO, OpenCV ROI-only, and the safe hybrid with periodic full-frame YOLO.
Report FPS and end-to-end latency together with precision, recall, F1 and mAP on
labelled frames. On no-sign video, report false detections per minute. A raw count of
frames containing any output is coverage, not accuracy. Select the deployment mode
and confidence threshold from these measurements on the actual laptop.

The active training baseline is pretrained **YOLO26s at `imgsz=640`**. Dataset
preparation must not be changed merely to make a larger model appear better:
YOLO26n is only a latency fallback, while YOLO26m is tested only after the dataset is
large and balanced enough and a GPU deployment target is available. Keep the same
train/validation/test split for every model comparison.
