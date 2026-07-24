# MYSignVoice: Dataset & Manual Labeling Guide

If your YOLOv8 model has low accuracy, the most common reason is **not enough varied data**. Training a deep learning model is like teaching a child: if you only show them perfect, sunny pictures of a Stop sign, they won't recognize a Stop sign at night or in the rain. 

This guide explains how to manually label images, use existing Roboflow datasets, combine multiple datasets, and retrain in Google Colab.

---

## 1. How Many Classes Are We Training?
According to our `dataset/data.yaml` file, we are currently targeting **50 distinct traffic sign classes**. 

These include:
*   **Speed Limits** (`speed_5` through `speed_110`)
*   **Prohibitory Signs** (`no_entry`, `no_uturn`, `no_parking`, `stop`)
*   **Warning Signs** (`school_zone`, `pedestrian_crossing`, `sharp_bend_left`)
*   **Mandatory / Directional** (`keep_left`, `turn_right`, `roundabout_direction`)

*(Note: Since you currently have a small dataset of about 84 images, targeting all 50 classes is extremely difficult. It is highly recommended to reduce the target to just the 20+ classes actually present in those 84 images. You can update your `data.yaml` to only include these 20+ classes, which will immediately improve your accuracy metrics.)*

---

## 2. Recommended Labeling Tool: Roboflow
To label your own dataset, we highly recommend using **[Roboflow](https://roboflow.com/)**. It is free for public projects, extremely user-friendly, and most importantly, it allows you to **export directly in YOLOv8 format**.

*(Alternative: If you want a completely offline, account-free tool, use [MakeSense.ai](https://www.makesense.ai/).)*

---

## 3. Step-by-Step Manual Labeling Process (If Starting From Scratch)

### Step 1: Collect Raw Images
*   Gather images of Malaysian traffic signs. 
*   **Important:** Do not just use perfect images from Google! Take photos with your phone. Include images where the sign is far away, blurry, at an angle, or partially hidden by a tree. 

### Step 2: Upload to Roboflow
*   Create a free account on Roboflow and create a new project. 
*   Set the project type to **Object Detection**.
*   Upload all your raw images.

### Step 3: Draw Bounding Boxes (Annotation)
*   Go through your images one by one.
*   Draw a tight rectangle (bounding box) around the traffic sign. 
*   Select the correct class name (e.g., `school_zone`). The names must perfectly match what you wrote in `data.yaml`.
*   **Rule of Thumb:** If the sign is too small or blurry for a human to read, do not label it. If you can't read it, the AI can't either!

### Step 4: Dataset Generation & Augmentation (The Secret to High Accuracy)
Since your dataset is small (e.g., only 300 images), you must use **Data Augmentation**. Roboflow can take your 300 images and artificially turn them into 900 images by applying filters.
In Roboflow, go to the "Generate" tab and add these augmentations:
1.  **Brightness:** +/- 25% (Simulates daytime and nighttime).
2.  **Blur:** Up to 1.5px (Simulates motion blur from a walking pedestrian).
3.  **Noise:** Up to 2% (Simulates cheap camera sensors).
4.  **Rotation:** +/- 10° (Simulates holding the phone at a tilted angle).

### Step 5: Export to YOLOv8
*   Click **Generate Version**.
*   Click **Export Dataset**.
*   Select **YOLOv8** format.
*   Roboflow will give you a Python snippet or a download link. 

### Step 6: Retrain in Google Colab
*   Paste the Roboflow download link into your Google Colab notebook to download the newly labelled and augmented dataset.
*   Run your YOLOv8 training command again.

---

## 4. Using an Existing Roboflow Universe Dataset (9,483 Images)

Roboflow Universe is a public marketplace where other people share pre-labelled datasets. If you find a traffic sign dataset with 9,483 images that is already labelled, you can use it directly without manually drawing any bounding boxes.

### Step 1: Fork the Dataset
*   Go to the dataset page on Roboflow Universe.
*   Click **"Download this Dataset"** or **"Fork"** to copy it into your own Roboflow workspace.

### Step 2: Check the Class Names
*   Open the dataset in your workspace and look at the class list.
*   Write down every class name (e.g., the Roboflow dataset might use names like `Speed limit (60km/h)` or `prohibitory` instead of our `speed_60` or `no_entry`).

### Step 3: Export to YOLOv8 for Colab
*   In Roboflow, click **Generate** > **Export** > **YOLOv8** format.
*   Choose **"show download code"** to get the Python snippet.
*   The snippet looks like this:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
version = project.version(1)
dataset = version.download("yolov8")
```
*   Paste this snippet into your Colab notebook. It will download the full 9,483-image dataset directly into Colab.

---

## 5. Combining Multiple Datasets (The Class Name Problem)

You have two datasets with **different class names and different label index numbers**:

| Dataset | Source | Size | Example Class Names |
| :--- | :--- | :--- | :--- |
| **Dataset A** | Your previous MTSD training (Google Drive, `best.pt`) | ~1,000 images | Uses MTSD class IDs (e.g., `regulatory--no-entry--g1`) |
| **Dataset B** | The Roboflow Universe dataset | ~9,483 images | Uses Roboflow class names (e.g., `No Entry`, `Speed Limit 60`) |

These two datasets use **different class indexes** in their YOLO `.txt` label files. For example, "No Entry" might be class `11` in Dataset A but class `3` in Dataset B. If you just dump them into the same folder, the AI will get completely confused.

### How to Combine Them: The Class Remapping Process

#### Step 1: Decide on ONE Master Class List
Create a single unified `data.yaml` with the final class list you want. For example, if you only target 20 classes:
```yaml
nc: 20
names:
  0: speed_30
  1: speed_50
  2: speed_60
  3: speed_80
  4: no_entry
  5: no_uturn
  6: stop
  7: give_way
  8: pedestrian_crossing
  9: school_zone
  # ... etc
```

#### Step 2: Create a Remapping Dictionary
For each dataset, create a Python dictionary that maps their old class index to your new master index.
```python
# Example: Dataset A (MTSD) remapping
remap_A = {
    0: 2,   # MTSD class 0 (speed_60) -> Master class 2
    5: 4,   # MTSD class 5 (no_entry) -> Master class 4
    # ... map all classes
}

# Example: Dataset B (Roboflow) remapping
remap_B = {
    0: 6,   # Roboflow class 0 (Stop) -> Master class 6
    3: 4,   # Roboflow class 3 (No Entry) -> Master class 4
    # ... map all classes
}
```

#### Step 3: Run a Remapping Script
Use a Python script to go through every `.txt` label file in each dataset and replace the old class index with the new one. Here is a ready-to-use script:

```python
import os

def remap_labels(label_dir, remap_dict, output_dir):
    """Remap YOLO label files from old class IDs to new master IDs."""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, filename), 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            old_class = int(parts[0])
            if old_class in remap_dict:
                parts[0] = str(remap_dict[old_class])
                new_lines.append(' '.join(parts) + '\n')
            # If old_class is NOT in remap_dict, we skip that label
            # (it means we don't need that class)
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.writelines(new_lines)
    print(f"Remapped {label_dir} -> {output_dir}")
```

#### Step 4: Merge Into One Folder
After remapping, copy all images and their remapped labels into one combined folder:
```
combined_dataset/
  train/
    images/    <-- All images from both datasets
    labels/    <-- All remapped .txt files from both datasets
  val/
    images/
    labels/
```

---

## 6. Training in Google Colab (Where to Save Everything)

### Save on Google Drive, NOT on Colab Local Storage
Colab's local storage (`/content/`) gets **wiped every time your session disconnects** (usually after 90 minutes of idle or 12 hours max). If you save your trained model there, it will be gone forever.

### Recommended Colab Workflow

```python
# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Set your project folder on Drive
PROJECT_DIR = '/content/drive/MyDrive/TrafficSignProject'

# Step 3: Download the Roboflow dataset (saves to Colab local, which is fast)
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("your-project")
version = project.version(1)
dataset = version.download("yolov8")

# Step 4: Train YOLOv8 (output saves to local first for speed)
!yolo train model=yolov8n.pt data={dataset.location}/data.yaml epochs=100 imgsz=640 batch=16

# Step 5: Copy the trained model to Google Drive (IMPORTANT!)
import shutil
shutil.copytree(
    '/content/runs/detect/train/weights',
    f'{PROJECT_DIR}/training_runs/v2_combined/weights',
    dirs_exist_ok=True
)
print("Model saved to Google Drive!")

# Step 6: Also save the training metrics
shutil.copytree(
    '/content/runs/detect/train',
    f'{PROJECT_DIR}/training_runs/v2_combined/full_results',
    dirs_exist_ok=True
)
```

### Why This Workflow?
| Step | Location | Why |
| :--- | :--- | :--- |
| Download dataset | Colab local (`/content/`) | Faster I/O during training |
| Train model | Colab local (`/content/runs/`) | GPU training needs fast disk |
| Save final `best.pt` | Google Drive (`/content/drive/`) | Persistent, survives session restart |
| Download `best.pt` to laptop | Your PC `dataset/best.pt` | For webcam testing and Android deployment |

---

## 7. Quick Decision: What Should You Do Right Now?

| Option | Effort | Expected Accuracy |
| :--- | :--- | :--- |
| **Option A: Just use the Roboflow 9,483 dataset** | Low (1 hour) | Good, but class names might not match Malaysian signs exactly |
| **Option B: Combine Roboflow + your old MTSD data** | Medium (3-4 hours) | Better, more variety in training data |
| **Option C: Option B + manually label the 84 test images** | High (1-2 days) | Best, because the model has seen the exact sign styles it will be tested on |

**Recommendation:** Start with **Option A** to get a working demo quickly. If accuracy is still low on the 84 test images, move to **Option C** by manually labelling the 84 test images on Roboflow and adding them to the combined dataset. Since the final project is graded on how many of those 84 signs you detect correctly (70 marks), having those exact images in your training set will give you the highest score.
