# 🚀 YOLOv8 Training with Roboflow Dataset — Complete Colab Guide

> **Where**: Everything happens on **Google Colab** (free GPU)
> **Dataset**: Malaysia Road Sign Dataset from Roboflow (9,483 images)
> **Time**: ~45 minutes total

---

## Setup: Create a New Colab Notebook

1. Go to **https://colab.research.google.com**
2. Click **"New notebook"**
3. **Runtime** → **Change runtime type** → Select **T4 GPU** → **Save**
4. Rename notebook to: `MYSignVoice_Roboflow_Training`

---

## Cell 1 — Verify GPU is Working

```python
# ============================================================
# CELL 1: Verify GPU
# ============================================================
!nvidia-smi
```

**Expected output:** You should see "Tesla T4" in the output.

---

## Cell 2 — Install Dependencies

```python
# ============================================================
# CELL 2: Install ultralytics and roboflow
# ============================================================
!pip install ultralytics roboflow -q
print("✅ Installation complete!")
```

---

## Cell 3 — Mount Google Drive (to save your model)

```python
# ============================================================
# CELL 3: Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
save_dir = "/content/drive/MyDrive/TrafficSignProject"
os.makedirs(save_dir, exist_ok=True)
print(f"✅ Google Drive mounted! Model will be saved to: {save_dir}")
```

**Action:** A popup will appear asking you to sign in — click **Allow**.

---

## Cell 4 — Download the Roboflow Dataset

```python
# ============================================================
# CELL 4: Download Malaysia Road Sign Dataset from Roboflow
# ============================================================
import os

dataset_location = "/content/dataset"
os.makedirs(dataset_location, exist_ok=True)

# Change directory, download, unzip, and remove the zip
%cd {dataset_location}
!curl -L "https://app.roboflow.com/ds/U63mKYkItS?key=0oIpI21top" > roboflow.zip; unzip -q roboflow.zip; rm roboflow.zip
%cd /content

print(f"\n✅ Dataset downloaded to: {dataset_location}")
```

**Expected output:** The dataset will download to `/content/dataset/`

---

## Cell 5 — Inspect the Dataset (IMPORTANT — send me this output!)

```python
# ============================================================
# CELL 5: Inspect dataset structure and classes
# ============================================================
import yaml

# Read the data.yaml file
data_yaml_path = f"{dataset_location}/data.yaml"
with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Dataset path: {dataset_location}")
print(f"Number of classes: {data_config.get('nc', 'unknown')}")
print(f"\nClass names:")
for i, name in enumerate(data_config.get('names', [])):
    print(f"  [{i}] {name}")

# Count images in each split
for split in ['train', 'valid', 'test']:
    img_dir = os.path.join(dataset_location, split, 'images')
    if os.path.exists(img_dir):
        count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"\n{split}: {count} images")
    else:
        print(f"\n{split}: folder not found")

print("=" * 60)
```

> [!IMPORTANT]
> **SEND ME THE OUTPUT OF THIS CELL!** I need to see:
> 1. How many classes there are
> 2. What the class names are
> 3. How many train/val/test images
> This tells me if the dataset is good enough or if we need adjustments.

---

## Cell 6 — Preview Some Training Images

```python
# ============================================================
# CELL 6: Preview a few training images with their labels
# ============================================================
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

train_img_dir = os.path.join(dataset_location, 'train', 'images')
train_lbl_dir = os.path.join(dataset_location, 'train', 'labels')

# Get class names
class_names = data_config.get('names', [])

# Show 6 random images
img_files = sorted(os.listdir(train_img_dir))[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Sample Training Images", fontsize=16)

for idx, img_file in enumerate(img_files):
    ax = axes[idx // 3][idx % 3]
    img_path = os.path.join(train_img_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read corresponding label
    lbl_file = os.path.splitext(img_file)[0] + '.txt'
    lbl_path = os.path.join(train_lbl_dir, lbl_file)

    title = img_file
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        if lines:
            cls_id = int(lines[0].split()[0])
            if cls_id < len(class_names):
                title = f"{img_file}\nClass: {class_names[cls_id]}"

    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "sample_training_images.png"), dpi=150)
plt.show()
print("✅ Preview saved to Google Drive")
```

---

## Cell 7 — Train YOLOv8-nano

```python
# ============================================================
# CELL 7: TRAIN YOLOv8-nano
# This is the main training cell. Takes ~20-40 minutes.
# ============================================================
from ultralytics import YOLO

# Load pretrained YOLOv8-nano
model = YOLO('yolov8n.pt')

# Train on the Roboflow dataset
results = model.train(
    data=f"{dataset_location}/data.yaml",
    epochs=100,              # 100 training rounds
    imgsz=640,               # Input image size
    batch=16,                # Images per batch (reduce to 8 if CUDA out of memory)
    patience=20,             # Stop early if no improvement for 20 epochs
    project=save_dir,        # Save to Google Drive
    name='roboflow_v1',      # Run name
    exist_ok=True,           # Overwrite if exists
    plots=True,              # Generate training plots
    save=True,               # Save checkpoints
    verbose=True             # Show progress
)

print("\n✅ Training complete!")
print(f"Best model saved at: {save_dir}/roboflow_v1/weights/best.pt")
```

> [!TIP]
> If you get **"CUDA out of memory"**, change `batch=16` to `batch=8`.
> Training takes about 20-40 minutes. You can watch the progress in real-time.

---

## Cell 8 — View Training Results

```python
# ============================================================
# CELL 8: View training results
# ============================================================
from IPython.display import Image, display

results_dir = f"{save_dir}/roboflow_v1"

# Show the confusion matrix
confusion_path = f"{results_dir}/confusion_matrix.png"
if os.path.exists(confusion_path):
    print("📊 Confusion Matrix:")
    display(Image(filename=confusion_path, width=800))

# Show the results plot (loss curves, mAP)
results_path = f"{results_dir}/results.png"
if os.path.exists(results_path):
    print("\n📈 Training Curves:")
    display(Image(filename=results_path, width=800))

# Show validation predictions
val_pred_path = f"{results_dir}/val_batch0_pred.png"
if os.path.exists(val_pred_path):
    print("\n🔍 Validation Predictions:")
    display(Image(filename=val_pred_path, width=800))
```

---

## Cell 9 — Test on Sample Images

```python
# ============================================================
# CELL 9: Quick test — run detection on validation images
# ============================================================
best_model = YOLO(f"{save_dir}/roboflow_v1/weights/best.pt")

# Run validation to get metrics
metrics = best_model.val(data=f"{dataset_location}/data.yaml")

print("\n" + "=" * 60)
print("FINAL MODEL METRICS")
print("=" * 60)
print(f"  mAP@50:     {metrics.box.map50:.1%}")
print(f"  mAP@50-95:  {metrics.box.map:.1%}")
print(f"  Precision:   {metrics.box.mp:.1%}")
print(f"  Recall:      {metrics.box.mr:.1%}")
print("=" * 60)
```

---

## Cell 10 — Export to ONNX (for Android later)

```python
# ============================================================
# CELL 10: Export model to ONNX format (for future Android use)
# ============================================================
import shutil

best_pt = f"{save_dir}/roboflow_v1/weights/best.pt"
model = YOLO(best_pt)

# Export to ONNX
model.export(format='onnx', imgsz=640, simplify=True)
print("✅ ONNX export complete!")

# Also copy best.pt to an easy-to-find location
shutil.copy(best_pt, f"{save_dir}/best.pt")
print(f"✅ best.pt copied to: {save_dir}/best.pt")
print("\n📥 Download best.pt from Google Drive to your laptop!")
print("   Path: Google Drive > TrafficSignProject > best.pt")
```

---

## After Training — What To Do on Your Laptop

1. **Download `best.pt`** from Google Drive → `TrafficSignProject/best.pt`
2. **Copy** `best.pt` into: `preliminary/member4_shape_detection/`
3. **Run the test script:**
   ```
   cd preliminary/member4_shape_detection
   pip install ultralytics opencv-python
   python test_84_images.py
   ```
4. **Run the webcam demo:**
   ```
   python webcam_yolo_demo.py
   ```
5. **Take screenshots** for your report and presentation!
