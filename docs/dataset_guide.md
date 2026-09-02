# MYSignVoice Dataset Collection and Management Guide

This is the active collection guide for the 63-class YOLO26s web system. The objective is not merely a large image count: it is a balanced set of correctly boxed signs, realistic laptop-camera scenes, and hard negatives that are independent across train, validation, and test splits.

## 1. Class contract

`dataset/data.yaml` is authoritative. Roboflow must contain the same 63 class names in the same order. Keep the lowercase dash-separated names; do not change them to underscores after annotation starts. Class 32 is `pass-obstacle-on-either-side`; IDs 47–62 are the approved Malaysian-road-sign expansion.

Every bounding box should tightly cover the complete visible sign face. In any retained image, label every visible instance that belongs to one of the 63 target classes. A missing box teaches the detector that a real target can be background.

## 2. Fast collection methods

### Record local video

Recording 1080p road or printed-sign video is the fastest way to gather the exact camera conditions used by the web app. Extract candidate frames at about 1–2 frames per second, then remove near-duplicates. Capture variation in:

- distance and sign size;
- daylight, shade, glare, low light, and rain;
- front, left/right angle, tilt, and partial occlusion;
- motion blur and camera focus;
- cluttered red, blue, and yellow backgrounds;
- several signs in the same frame;
- frames with no traffic sign.

Do not place adjacent frames from one video into different dataset splits. Group by video/location/session first, then assign the whole group to train, validation, or test. Otherwise nearly identical frames will inflate validation results.

### Reuse public datasets carefully

You may reuse another detection dataset even when its original class list differs, subject to its licence:

1. Keep only annotations whose visual sign and meaning unambiguously match one canonical MYSignVoice class.
2. Remap that source class to the exact canonical class name.
3. Delete incorrect or irrelevant boxes; never force a similar-looking sign into the nearest class.
4. Inspect imported boxes manually because label quality and box conventions vary.
5. Keep source attribution and licence information with the imported data.

If an image contains a canonical target sign, it must be boxed even when the original dataset did not label it. A truly out-of-scope sign can remain background, but exclude ambiguous cases that are visually indistinguishable from a target class. For a genuine no-sign image, use an empty YOLO label file.

Public sources can accelerate common classes, but Malaysian signs and laptop-camera footage are still required for the target domain.

### Approved Malaysia road sign source mapping

The Roboflow workspace's **Malaysia road sign dataset** has compatible
annotations for the following approved additions. Preserve the boxes, but map the
source label to the exact target label before it enters a MYSignVoice dataset
version:

| Source label | MYSignVoice label |
|---|---|
| Bumps | `bumps-warning` |
| Bus stop | `bus-stop` |
| Camera operation zone | `camera-operation-zone` |
| Cow nearby | `cow-nearby-warning` |
| Height limit | `height-limit` |
| No parking | `no-parking` |
| Parking area | `parking-area` |
| Towing area | `towing-area` |
| Chevron (left) | `chevron-left` |
| Chevron (right) | `chevron-right` |
| Crossroad on the left | `crossroad-left-warning` |
| Crossroad on the right | `crossroad-right-warning` |
| Road narrows on left | `road-narrows-left-warning` |
| Road narrows on right | `road-narrows-right-warning` |
| Roadway diverges | `roadway-diverges-warning` |
| Reverse turn | `reverse-turn-warning` |

Do not import that dataset's generic `Speed limit` class: it does not identify
the numeric limit and cannot safely map to the existing speed-limit labels.
When a selected image also contains another target sign, label or map that sign
too; otherwise remove the image from the import set. Do not retain source-only
annotations under their original names, because that would create an unintended
64th class or a mislabeled training box.

### Targeted image search

Use exact queries for underrepresented classes and Malaysian variants. Verify reuse permission, remove duplicates/watermarks where necessary, and avoid filling the test set with internet images seen during collection decisions.

## 3. Practical data targets

Use staged targets rather than waiting for a perfect dataset:

- Pilot: enough correctly labelled examples to verify all 63 class IDs and complete one training run.
- Baseline: aim for at least dozens of diverse real instances per class, with more for classes that are visually similar or frequently missed.
- Improvement rounds: collect examples from actual failure cases instead of applying ever-stronger augmentation.

The existing 84 images are useful seeds, but they cannot establish reliable 63-class performance. Augmented copies do not count as independent real examples.

## 4. Image quality and storage

Keep the original high-resolution source files outside Git. For working copies, preserve aspect ratio and enough detail for distant signs; a maximum long side of 1280–1920 pixels with JPEG quality around 90 is a safer default than shrinking every image to 800 pixels. Ultralytics will letterbox inputs during training and inference.

```text
miniproject/
└── dataset/
    ├── raw_images/                  # Original local or licensed source images
    ├── annotated/
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── val/
    │   │   ├── images/
    │   │   └── labels/
    │   └── test/
    │       ├── images/
    │       └── labels/
    └── data.yaml                    # Locked 63-class Ultralytics configuration
```

Store versioned Roboflow exports or compressed dataset archives in Google Drive/OneDrive, not Git. In Colab, copy the dataset archive to the VM disk before training to avoid slow per-file reads from mounted Drive.

## 5. Split rules

Use approximately 70–80% train, 10–20% validation, and 10–15% test, while keeping every class represented where possible. More important than the exact percentages:

- group images by source video, physical sign, location, and capture session;
- place each group in only one split;
- reserve the test set and do not use it to choose augmentation or thresholds;
- include no-sign and hard-negative scenes in validation and test;
- check class and object-instance counts, not only image counts.

## 6. Augmentation

Apply augmentation only to the training split. Start with the modest settings in `training/train_colab.py`: small brightness/saturation changes, translation, scale, rotation, and limited mosaic. Do not use horizontal or vertical flips because direction is part of the class meaning. Add mild camera-like blur, compression, noise, and exposure variation only after reviewing samples; augmentation must not make the sign unreadable or change its meaning.

More varied original images and correct labels usually provide more value than aggressive augmentation.

## 7. Pre-training checklist

- `nc: 63` and all names/order match `dataset/data.yaml`.
- `pass-obstacle-on-either-side` is class ID 32.
- Every target sign in each retained image has a tight box.
- No image/label pair is missing, except intentional negatives with empty labels.
- No near-duplicate scene crosses train/validation/test.
- Each class has validation examples; rare-class gaps are documented.
- Camera images and hard-negative/no-sign scenes are included.
- A new immutable Roboflow dataset version is generated before the Colab run.

Then follow `docs/roboflow_training_guide.md` and train the default YOLO26s 640-pixel baseline. Model benchmarks cannot substitute for measurements on this held-out dataset and the actual laptop/server.
