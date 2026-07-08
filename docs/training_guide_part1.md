# 🚀 YOLOv8-nano Training Guide — Part 1: Dataset Preparation & First Training Run

> **Where**: Everything in Part 1 happens entirely on **Google Colab**. You don't need to touch your local PC at all.
>
> **What you need**: Your 6 RAR files already uploaded to Google Drive folder `TrafficSignProject/`
>
> **Time estimate**: ~30–45 minutes (including training)

---

## 📋 Overview of Part 1

| Step | What | Time |
|------|------|------|
| Step 1 | Create Colab notebook & set up GPU | 2 min |
| Step 2 | Mount Google Drive & extract all 6 RAR files | 5 min |
| Step 3 | Explore the dataset structure | 2 min |
| Step 4 | Convert MTSD ground truth → YOLO format | 5 min |
| Step 5 | Split into train/val sets (80/20) | 2 min |
| Step 6 | Create `data.yaml` config | 1 min |
| Step 7 | Train YOLOv8-nano | ~15–30 min |
| Step 8 | View results & save model to Drive | 3 min |

---

## Step 1: Create Colab Notebook & Enable GPU

1. Go to **https://colab.research.google.com**
2. Click **"New notebook"**
3. At the top menu: **Runtime** → **Change runtime type** → Select **T4 GPU** → Click **Save**
4. Rename the notebook (click the title at the top) to: **`MYSignVoice_Training_v1`**

> [!TIP]
> You can verify the GPU is working by running this cell:
> ```python
> !nvidia-smi
> ```
> You should see "Tesla T4" in the output.

---

## Step 2: Mount Google Drive & Extract RAR Files

### Cell 1 — Mount Drive

```python
# ============================================================
# CELL 1: Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted!")
```

> A popup will ask you to sign in and authorize — click **Allow**.

### Cell 2 — Install RAR extractor & extract all 6 files

```python
# ============================================================
# CELL 2: Extract all 6 RAR files
# ============================================================
!pip install patool pyunpack -q

import os
from pyunpack import Archive

# === CHANGE THIS to match your Drive folder ===
drive_folder = "/content/drive/MyDrive/TrafficSignProject"
extract_to = "/content/mtsd_raw"

os.makedirs(extract_to, exist_ok=True)

# Find all .rar files in your Drive folder
rar_files = sorted([f for f in os.listdir(drive_folder) if f.lower().endswith('.rar')])
print(f"Found {len(rar_files)} RAR files: {rar_files}")

# Extract each one
for rar in rar_files:
    rar_path = os.path.join(drive_folder, rar)
    print(f"\n📦 Extracting: {rar} ...")
    try:
        Archive(rar_path).extractall(extract_to)
        print(f"   ✅ Done: {rar}")
    except Exception as e:
        print(f"   ❌ Error with {rar}: {e}")

print(f"\n🎉 All extractions complete! Files in: {extract_to}")
```

> [!NOTE]
> If `pyunpack` has trouble with RAR files, try this alternative extraction method instead:
> ```python
> # Alternative: Install unrar and use it directly
> !apt-get install -y unrar -qq
> 
> import os
> drive_folder = "/content/drive/MyDrive/TrafficSignProject"
> extract_to = "/content/mtsd_raw"
> os.makedirs(extract_to, exist_ok=True)
> 
> for f in sorted(os.listdir(drive_folder)):
>     if f.lower().endswith('.rar'):
>         print(f"📦 Extracting {f}...")
>         !unrar x -o+ "{drive_folder}/{f}" "{extract_to}/"
>         print(f"   ✅ Done")
> ```

---

## Step 3: Explore What's Inside

### Cell 3 — List the extracted contents

```python
# ============================================================
# CELL 3: Explore the dataset structure
# ============================================================
import os

extract_to = "/content/mtsd_raw"

print("=" * 60)
print("TOP-LEVEL CONTENTS:")
print("=" * 60)
for item in sorted(os.listdir(extract_to)):
    full = os.path.join(extract_to, item)
    if os.path.isdir(full):
        count = sum(len(files) for _, _, files in os.walk(full))
        print(f"  📁 {item}/  ({count} files inside)")
    else:
        size_mb = os.path.getsize(full) / (1024*1024)
        print(f"  📄 {item}  ({size_mb:.1f} MB)")

# Look for ground truth files
print("\n" + "=" * 60)
print("SEARCHING FOR GROUND TRUTH FILES (*.txt):")
print("=" * 60)
for root, dirs, files in os.walk(extract_to):
    for f in files:
        if f.endswith('.txt'):
            filepath = os.path.join(root, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  📝 {filepath}  ({size_kb:.1f} KB)")

# Count images in Detection and Recognition folders
print("\n" + "=" * 60)
print("IMAGE COUNTS:")
print("=" * 60)
img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
for root, dirs, files in os.walk(extract_to):
    imgs = [f for f in files if f.lower().endswith(img_exts)]
    if imgs:
        rel = os.path.relpath(root, extract_to)
        print(f"  📁 {rel}: {len(imgs)} images")
```

> [!IMPORTANT]
> **Read the output carefully!** Based on the MTSD paper you shared, you should see:
> - A `Detection/` folder with ~1000 scene images (large images with signs in them)
> - A `Recognition/` folder with ~2056 cropped sign images (small 32×32 images of individual signs)
> - Ground truth files: `GT_Detection.txt`, `GT_Recognition.txt`, `Statistics_GT.txt`, `GT.txt`
>
> **For YOLOv8 training, we use the `Detection/` folder + `GT_Detection.txt`** (not the Recognition folder, because YOLO needs full scene images with bounding boxes, not pre-cropped signs).

---

## Step 4: Convert MTSD Ground Truth → YOLO Format

> [!IMPORTANT]
> This is the most critical step. The MTSD dataset uses a **custom CSV-like format**:
> ```
> File Name;X;Y;Width;Height;TS Color;Shape;Class ID;Lightning;Image Source
> ```
> But YOLOv8 needs **one `.txt` file per image** in this format:
> ```
> class_id  x_center  y_center  width  height
> ```
> (all values normalized to 0–1 relative to image size)
>
> The script below does this conversion automatically.

### Cell 4 — Convert ground truth format

```python
# ============================================================
# CELL 4: Convert MTSD GT_Detection.txt → YOLO format labels
# ============================================================
import os
import csv
from PIL import Image

# === PATHS — Adjust these if your folder structure is different ===
extract_to = "/content/mtsd_raw"

# Try to find the ground truth and detection folder automatically
gt_detection_path = None
detection_images_dir = None

for root, dirs, files in os.walk(extract_to):
    for f in files:
        if f == "GT_Detection.txt":
            gt_detection_path = os.path.join(root, f)
        if f == "GT.txt" and gt_detection_path is None:
            # Fallback: use GT.txt which has all info
            pass
    for d in dirs:
        if d.lower() == "detection":
            detection_images_dir = os.path.join(root, d)

print(f"📝 GT file found: {gt_detection_path}")
print(f"📁 Detection images folder: {detection_images_dir}")

# If not found automatically, set manually:
# gt_detection_path = "/content/mtsd_raw/GT_Detection.txt"
# detection_images_dir = "/content/mtsd_raw/Detection"

assert gt_detection_path is not None, "❌ Could not find GT_Detection.txt! Check Step 3 output and set path manually."
assert detection_images_dir is not None, "❌ Could not find Detection folder! Check Step 3 output and set path manually."

# ---------------------------------------------------------------
# MTSD uses shape-based Class IDs (1-10 for shapes).
# But the actual sign classes are in the TS Class column.
# The "Class ID" column in GT_Detection.txt is the SHAPE ID (1-10):
#   1 = Red Circle, 2 = Directive No, 3 = Diamond,
#   4 = Rectangle, 5 = Blue Circle, 6 = Flip Triangle,
#   7 = Regulatory No, 8 = No Entry, 9 = Pentagon, 10 = Octagon
#
# For our YOLOv8 training, we will use these SHAPE-based classes
# as the class IDs. This gives us 10 categories which is a good
# starting point. Later you can refine to use full 66 sign classes.
#
# CLASS MAPPING (shape-based, 0-indexed for YOLO):
#   0: red_circle          (prohibitory signs like speed limits)
#   1: directive_no        (no-entry style regulatory)
#   2: diamond             (warning signs - yellow diamond)
#   3: rectangle           (guide/information signs - blue rect)
#   4: blue_circle         (mandatory signs)
#   5: flip_triangle       (give way - inverted triangle)
#   6: regulatory_no       (regulatory no signs)
#   7: no_entry            (no entry - specific)
#   8: pentagon            (school zone etc.)
#   9: octagon             (stop sign)
# ---------------------------------------------------------------

CLASS_NAMES = {
    1: "red_circle",
    2: "directive_no",
    3: "diamond",
    4: "rectangle",
    5: "blue_circle",
    6: "flip_triangle",
    7: "regulatory_no",
    8: "no_entry",
    9: "pentagon",
    10: "octagon",
}

# Output directory for YOLO-format labels
yolo_labels_dir = "/content/yolo_dataset/all_labels"
yolo_images_dir = "/content/yolo_dataset/all_images"
os.makedirs(yolo_labels_dir, exist_ok=True)
os.makedirs(yolo_images_dir, exist_ok=True)

# ---------------------------------------------------------------
# Parse GT_Detection.txt
# Format: File Name;X;Y;Width;Height;TS Color;Shape;Class ID;Lightning;Image Source
# ---------------------------------------------------------------
annotations = {}  # { filename: [ (class_id, x, y, w, h), ... ] }
skipped = 0

with open(gt_detection_path, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f, delimiter=';')
    header = next(reader)  # Skip header row
    print(f"\n📋 GT Header: {header}")

    for row in reader:
        if len(row) < 8:
            skipped += 1
            continue
        try:
            filename = row[0].strip().strip("'\"")
            x = int(row[1].strip())
            y = int(row[2].strip())
            w = int(row[3].strip())
            h = int(row[4].strip())
            # ts_color = row[5].strip()
            # shape = row[6].strip()
            class_id = int(row[7].strip())

            if class_id not in CLASS_NAMES:
                skipped += 1
                continue

            if filename not in annotations:
                annotations[filename] = []
            annotations[filename].append((class_id, x, y, w, h))
        except (ValueError, IndexError) as e:
            skipped += 1
            continue

print(f"\n📊 Parsed annotations for {len(annotations)} images ({skipped} rows skipped)")

# ---------------------------------------------------------------
# Convert to YOLO format and copy images
# ---------------------------------------------------------------
import shutil

converted = 0
errors = 0

for img_filename, bboxes in annotations.items():
    # Find the image file
    img_path = os.path.join(detection_images_dir, img_filename)
    if not os.path.exists(img_path):
        # Try common variations
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            base = os.path.splitext(img_filename)[0]
            alt_path = os.path.join(detection_images_dir, base + ext)
            if os.path.exists(alt_path):
                img_path = alt_path
                break
        else:
            errors += 1
            continue

    # Get actual image dimensions
    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    except Exception:
        errors += 1
        continue

    # Create YOLO label file
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(yolo_labels_dir, label_filename)

    with open(label_path, 'w') as lf:
        for (cls_id, bx, by, bw, bh) in bboxes:
            # Convert MTSD format (top-left x, y, width, height) to
            # YOLO format (center_x, center_y, width, height) normalized
            yolo_class = cls_id - 1  # YOLO uses 0-indexed classes

            center_x = (bx + bw / 2.0) / img_w
            center_y = (by + bh / 2.0) / img_h
            norm_w = bw / img_w
            norm_h = bh / img_h

            # Clamp to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))

            lf.write(f"{yolo_class} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    # Copy image to YOLO images directory
    dst_img = os.path.join(yolo_images_dir, img_filename)
    if not os.path.exists(dst_img):
        shutil.copy2(img_path, dst_img)

    converted += 1

print(f"\n✅ Converted {converted} images to YOLO format")
print(f"❌ Skipped {errors} images (not found or unreadable)")
print(f"📁 YOLO labels → {yolo_labels_dir}")
print(f"📁 YOLO images → {yolo_images_dir}")
```

> [!WARNING]
> **If the script says "Could not find GT_Detection.txt" or "Could not find Detection folder":**
> Go back to Step 3's output, find the actual paths, and manually set them in the two lines marked with `# If not found automatically, set manually:`.

---

## Step 5: Split into Train/Val (80/20)

### Cell 5 — Split dataset

```python
# ============================================================
# CELL 5: Split dataset into train (80%) and val (20%)
# ============================================================
import os
import random
import shutil

random.seed(42)  # For reproducibility

yolo_images_dir = "/content/yolo_dataset/all_images"
yolo_labels_dir = "/content/yolo_dataset/all_labels"

# Create train/val directories
for split in ['train', 'val']:
    os.makedirs(f"/content/yolo_dataset/{split}/images", exist_ok=True)
    os.makedirs(f"/content/yolo_dataset/{split}/labels", exist_ok=True)

# Get all image files that have matching label files
img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
all_images = [f for f in os.listdir(yolo_images_dir) if f.lower().endswith(img_exts)]

# Only keep images that have a corresponding label file
paired = []
for img in all_images:
    label = os.path.splitext(img)[0] + ".txt"
    if os.path.exists(os.path.join(yolo_labels_dir, label)):
        paired.append(img)

print(f"📊 Total image-label pairs: {len(paired)}")

# Shuffle and split
random.shuffle(paired)
split_idx = int(len(paired) * 0.8)
train_files = paired[:split_idx]
val_files = paired[split_idx:]

print(f"   🏋️ Train: {len(train_files)} images")
print(f"   📋 Val:   {len(val_files)} images")

# Copy files
for split_name, file_list in [('train', train_files), ('val', val_files)]:
    for img_file in file_list:
        label_file = os.path.splitext(img_file)[0] + ".txt"

        # Copy image
        shutil.copy2(
            os.path.join(yolo_images_dir, img_file),
            f"/content/yolo_dataset/{split_name}/images/{img_file}"
        )
        # Copy label
        shutil.copy2(
            os.path.join(yolo_labels_dir, label_file),
            f"/content/yolo_dataset/{split_name}/labels/{label_file}"
        )

print("\n✅ Dataset split complete!")
print(f"   Train images: /content/yolo_dataset/train/images/")
print(f"   Train labels: /content/yolo_dataset/train/labels/")
print(f"   Val images:   /content/yolo_dataset/val/images/")
print(f"   Val labels:   /content/yolo_dataset/val/labels/")
```

---

## Step 6: Create `data.yaml` Config

### Cell 6 — Write the YOLO config file

```python
# ============================================================
# CELL 6: Create data.yaml for YOLOv8 training
# ============================================================

yaml_content = """# MYSignVoice — Malaysian Traffic Sign Dataset (MTSD)
# YOLOv8 Training Configuration
# Shape-based classes from GT_Detection.txt (10 classes)

path: /content/yolo_dataset
train: train/images
val: val/images

nc: 10

names:
  0: red_circle
  1: directive_no
  2: diamond
  3: rectangle
  4: blue_circle
  5: flip_triangle
  6: regulatory_no
  7: no_entry
  8: pentagon
  9: octagon
"""

with open("/content/yolo_dataset/data.yaml", "w") as f:
    f.write(yaml_content)

print("✅ data.yaml created at /content/yolo_dataset/data.yaml")
print("\nContents:")
print(yaml_content)
```

> [!NOTE]
> **About the 10 classes**: The MTSD `GT_Detection.txt` uses **shape-based class IDs** (1–10), not the full 66 sign categories. This means:
> - Class 0 (red_circle) = All prohibitory signs (speed limits, no parking, etc.)
> - Class 2 (diamond) = All warning signs (curves, humps, crossings, etc.)
> - Class 9 (octagon) = Stop sign
>
> This is a **good starting point** for your first training run. In **Part 2** (later), we can refine to use the full 66 classes from `GT_Recognition.txt` + `GT.txt` for finer classification.

---

## Step 7: Train YOLOv8-nano! 🚀

### Cell 7 — Install Ultralytics & Train

```python
# ============================================================
# CELL 7: Install Ultralytics and train YOLOv8-nano
# ============================================================
!pip install ultralytics -q

from ultralytics import YOLO

# Load pretrained YOLOv8-nano (downloads automatically)
model = YOLO('yolov8n.pt')

# Train on our Malaysian Traffic Sign dataset
results = model.train(
    data='/content/yolo_dataset/data.yaml',
    epochs=100,            # Max 100 epochs
    imgsz=640,             # Input size 640x640
    batch=16,              # Batch size (T4 can handle 16 easily)
    patience=20,           # Stop early if no improvement for 20 epochs
    project='/content/drive/MyDrive/TrafficSignProject/training_runs',
    name='mtsd_v1',        # Run name
    exist_ok=True,         # Overwrite if exists
    plots=True,            # Generate training plots
    verbose=True,          # Show detailed output
    # --- Data augmentation (good defaults) ---
    hsv_h=0.015,           # Hue shift
    hsv_s=0.7,             # Saturation shift
    hsv_v=0.4,             # Value (brightness) shift
    degrees=10,            # Rotation up to 10°
    translate=0.1,         # Translation
    scale=0.5,             # Scale augmentation
    flipud=0.0,            # No vertical flip (signs shouldn't be upside down)
    fliplr=0.5,            # Horizontal flip 50%
    mosaic=1.0,            # Mosaic augmentation
)

print("\n🎉 TRAINING COMPLETE!")
```

> [!IMPORTANT]
> **Training will take approximately 15–30 minutes** on a free T4 GPU with ~1000 images. You'll see live output showing:
> - Current epoch
> - Box loss, class loss, DFL loss (should decrease)
> - mAP50 and mAP50-95 (should increase)
>
> **Don't close the browser tab!** Colab will disconnect if idle.

---

## Step 8: View Results & Save Model

### Cell 8 — Display training results

```python
# ============================================================
# CELL 8: View training results
# ============================================================
from IPython.display import Image, display
import os

results_dir = "/content/drive/MyDrive/TrafficSignProject/training_runs/mtsd_v1"

# Show training curves
print("📈 TRAINING CURVES:")
results_img = os.path.join(results_dir, "results.png")
if os.path.exists(results_img):
    display(Image(filename=results_img, width=900))
else:
    print("   results.png not found — check the results_dir path")

# Show confusion matrix
print("\n📊 CONFUSION MATRIX:")
cm_img = os.path.join(results_dir, "confusion_matrix.png")
if os.path.exists(cm_img):
    display(Image(filename=cm_img, width=700))

# Show sample predictions
print("\n🖼️ SAMPLE PREDICTIONS (Validation):")
val_img = os.path.join(results_dir, "val_batch0_pred.png")
if os.path.exists(val_img):
    display(Image(filename=val_img, width=900))
```

### Cell 9 — Check model file size and location

```python
# ============================================================
# CELL 9: Check saved model
# ============================================================
import os

results_dir = "/content/drive/MyDrive/TrafficSignProject/training_runs/mtsd_v1"
best_pt = os.path.join(results_dir, "weights", "best.pt")
last_pt = os.path.join(results_dir, "weights", "last.pt")

if os.path.exists(best_pt):
    size_mb = os.path.getsize(best_pt) / (1024 * 1024)
    print(f"✅ Best model saved: {best_pt}")
    print(f"   Size: {size_mb:.1f} MB")
else:
    print("❌ best.pt not found!")

if os.path.exists(last_pt):
    size_mb = os.path.getsize(last_pt) / (1024 * 1024)
    print(f"✅ Last model saved: {last_pt}")
    print(f"   Size: {size_mb:.1f} MB")

print(f"\n📁 All results saved to Google Drive at:")
print(f"   {results_dir}")
print(f"\n💡 The 'best.pt' file is your trained model.")
print(f"   In Part 2, we'll export it to ONNX → ncnn for Android.")
```

### Cell 10 — Quick test on a sample image

```python
# ============================================================
# CELL 10: Quick test — run detection on a sample image
# ============================================================
from ultralytics import YOLO
from IPython.display import Image, display
import os

results_dir = "/content/drive/MyDrive/TrafficSignProject/training_runs/mtsd_v1"
best_pt = os.path.join(results_dir, "weights", "best.pt")

# Load trained model
model = YOLO(best_pt)

# Pick a random image from validation set
val_images = "/content/yolo_dataset/val/images"
test_img = os.path.join(val_images, os.listdir(val_images)[0])

# Run inference
results = model.predict(
    source=test_img,
    conf=0.25,         # Confidence threshold
    save=True,
    project="/content/test_output",
    name="demo",
    exist_ok=True,
)

# Display the result
pred_img = f"/content/test_output/demo/{os.path.basename(test_img)}"
if os.path.exists(pred_img):
    print(f"🖼️ Detection result for: {os.path.basename(test_img)}")
    display(Image(filename=pred_img, width=800))
else:
    print("Prediction image not found at expected path. Check /content/test_output/")

# Print detections
for r in results:
    boxes = r.boxes
    print(f"\n📋 Found {len(boxes)} detections:")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_names = ['red_circle', 'directive_no', 'diamond', 'rectangle',
                       'blue_circle', 'flip_triangle', 'regulatory_no',
                       'no_entry', 'pentagon', 'octagon']
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"   • {name}: {conf:.1%} confidence")
```

---

## ✅ Part 1 Checklist — What You Should Have After This

- [ ] A trained `best.pt` model file saved on Google Drive at:
  `TrafficSignProject/training_runs/mtsd_v1/weights/best.pt`
- [ ] Training plots (loss curves, mAP curves) saved alongside
- [ ] Confusion matrix showing which sign shapes the model detects well
- [ ] A quick test showing the model can detect signs in a sample image

---

## 🔜 What Comes Next (Part 2 — Later)

| Part | What | When |
|------|------|------|
| **Part 2** | Export `best.pt` → ONNX → ncnn format for Android | After Part 1 works |
| **Part 3** | Refine training with full 66 classes (not just shapes) | After more labeling |
| **Part 4** | Data augmentation & model improvement | Before final demo |

---

## ⚠️ Troubleshooting

### "No labels found" error during training
→ This means the label `.txt` files don't match the image files. Go back to Cell 4 output and check how many images were converted. Make sure the image filenames in `GT_Detection.txt` match the actual files in `Detection/`.

### "CUDA out of memory" error
→ Reduce `batch` from `16` to `8` in Cell 7.

### Colab disconnects during training
→ Your model auto-saves to Google Drive every epoch. When reconnected:
```python
# Resume training from last checkpoint
from ultralytics import YOLO
model = YOLO('/content/drive/MyDrive/TrafficSignProject/training_runs/mtsd_v1/weights/last.pt')
model.train(resume=True)
```

### Very low mAP (<10%) after training
→ This usually means the label conversion went wrong. Run this debug cell:
```python
# Debug: Check a random label file
import os
label_dir = "/content/yolo_dataset/train/labels"
labels = os.listdir(label_dir)
if labels:
    sample = os.path.join(label_dir, labels[0])
    print(f"Sample label file: {labels[0]}")
    with open(sample) as f:
        print(f.read())
    print("\nExpected format: class_id center_x center_y width height")
    print("All values should be between 0 and 1 (except class_id which is 0-9)")
```
