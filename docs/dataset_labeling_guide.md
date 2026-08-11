# MYSignVoice — Dataset and Manual Labelling Guide

The project now has a reviewed **49-class, 84-image seed inventory**. The canonical
class names and YOLO IDs are in `dataset/data.yaml`; the annotated inventory workbook
is `class.xlsx`. Use the dedicated [49-Class Roboflow Annotation Plan](dataset_relabeling_guide.md)
for the current procedure and [Google Colab Training Guide](roboflow_training_guide.md)
for the active YOLO26s server-side training and deployment workflow. The annotation
format remains ordinary Ultralytics YOLO detection format; changing from YOLOv8 to
YOLO26 does not change any label ID or bounding-box text file.

> **Active deployment note:** this is a server-side web system, not an Android or
> iOS application. `docs/android_app_guide.md` and `training/convert_to_ncnn.py` are
> retained only as archival experiments and are not part of the current build or
> acceptance test.

## Non-negotiable rules

1. Label only the sign with a tight object-detection bounding box—not its pole or
   background.
2. Every class name must exactly match `dataset/data.yaml`, including spelling,
   hyphens, and order. IDs are zero-based; the corrected class ID 33 is
   `pass-obstacle-on-either-side`.
3. Add real, varied images for every class. The 84 supplied images alone contain
   only one to four instances of each class, which cannot support reliable
   49-class evaluation.
4. Split original images into train, validation, and test before augmentation.
   Augment training data only; never use horizontal/vertical flips for directional
   signs.
5. Use a test set that was not used for training when reporting accuracy.

## Combining an external dataset later

An external dataset may be useful only after checking that each sign’s visual meaning
matches the Malaysian sign class. Its YOLO numeric class IDs will almost certainly
differ from ours. Map its source names to this project’s canonical `data.yaml` list,
rewrite the label IDs, and preserve the source/license record before merging. Do not
merge folders with incompatible IDs directly.
