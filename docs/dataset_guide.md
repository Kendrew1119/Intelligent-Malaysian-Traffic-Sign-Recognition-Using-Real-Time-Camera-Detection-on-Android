# MYSignVoice — Traffic Sign Dataset Collection & Management Guide

This guide explains how to collect, organize, compress, and share a dataset of **8,000+ traffic sign images** for training the YOLOv8-nano model without running into storage limits or slowing down your Git repository.

---

## 1. How to Collect the Dataset

To reach a robust model, you need a diverse dataset of Malaysian traffic signs under different lighting (sunny, overcast, night), angles, and distances.

### Method A: Extracting Frames from Dashcam Videos (Fastest way to get thousands of images)
Instead of taking 8,000 individual photos, record 1080p videos on local roads or find driving videos on YouTube of Malaysian roads. 
Use a Python script with OpenCV to extract 1 frame every 1 or 2 seconds. This will generate thousands of high-quality, real-world images from different perspectives.

### Method B: Public Dataset Harvesting
Many international traffic signs look identical to Malaysian signs (following the UN Vienna Convention on Road Signs):
* **German Traffic Sign Detection Benchmark (GTSDB)**: High-quality circular prohibitory, blue mandatory, and red/yellow warning signs.
* **TT100K (Tsinghua-Tencent)**: Massive database containing many signs identical to Malaysia.
* **Kaggle**: Search for "Malaysian Traffic Sign Dataset" or "Traffic Sign Detection" to download pre-annotated subsets.

### Method C: Web Scraping
Use bulk downloader browser extensions (e.g., *Image Downloader* on Chrome) or Python scraper scripts to download images from Google Images for queries like:
* *"Malaysian speed limit sign"*
* *"Awas sign Malaysia"*
* *"Stop sign Malaysia"*
* *"No entry sign Malaysia"*

---

## 2. Where to Place the Data (Directory Structure)

Your local repository directory is structured to handle this dataset safely. The files inside the `dataset` folder are **already ignored by Git** (via `.gitignore`) so they will not be pushed to GitHub, keeping your repo lightweight.

Place your files in these exact folders locally:

```
miniproject/
└── dataset/
    ├── raw_images/                   # Put all newly collected, uncompressed photos here
    │   └── *.jpg / *.png
    ├── annotated/                    # Put your final labeled YOLO dataset here
    │   ├── train/
    │   │   ├── images/               # 80% of images for training
    │   │   └── labels/               # Matching .txt annotation files
    │   ├── val/
    │   │   ├── images/               # 10% of images for validation
    │   │   └── labels/
    │   └── test/
    │       ├── images/               # 10% of images for testing
    │       └── labels/
    ├── scripts/                      # Python helpers (organize, split, augment)
    └── data.yaml                     # YOLOv8 configuration file pointing to these folders
```

---

## 3. How to Manage Large Datasets (8,000 Images)

If you capture 8,000 photos on a modern phone (12MP - 48MP), the file size can exceed **10 GB to 30 GB**. This is too large to share, upload to Google Colab, or train efficiently.

Here is the 3-step strategy to handle this issue:

### Step 1: Compress and Resize (Reduces 10 GB down to ~300 MB)
YOLOv8-nano trains on **640x640 pixel** images. High-resolution images (like 4000x3000) slow down training and consume unnecessary space. 
We can resize raw images to a maximum width/height of **800px** and compress them as JPEG (quality = 85%). This keeps the features crystal clear for YOLO, but reduces each image's size from **~4 MB to ~40 KB** (a 100x reduction!).

Below is a Python script (`dataset/scripts/compress_images.py`) you can run to automate this.

### Step 2: Share via Cloud Storage (Google Drive / OneDrive)
1. Compress the resized `annotated` folder into a single `.zip` file (e.g., `dataset.zip`).
2. Upload this `.zip` file to **Google Drive** or **OneDrive**.
3. Share the folder/file link with your teammates so they can download it locally for coding.

### Step 3: Google Colab Integration (Fast I/O)
When training on Google Colab, **do not** train directly from Google Drive files, as this causes massive read latency (I/O bottleneck) that slows training down by 10x.
Instead, do this in your Colab notebook:
1. Mount Google Drive.
2. Copy the compressed `dataset.zip` from Drive to the temporary local Colab VM disk:
   `!cp /content/drive/MyDrive/dataset.zip /content/`
3. Unzip it directly on the local VM:
   `!unzip -q /content/dataset.zip -d /content/`
4. Train YOLO pointing to `/content/data.yaml`. This ensures blazing-fast training speeds.

---

## 🐍 Image Compression Script

Create and run this script to process all images in `dataset/raw_images/` before annotating them.

```python
# Save as: dataset/scripts/compress_images.py
import os
from PIL import Image

def compress_images(input_dir, output_dir, max_size=800, quality=85):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]
    
    print(f"Found {len(files)} images to compress...")
    
    for idx, filename in enumerate(files):
        img_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg") # save as jpg
        
        try:
            with Image.open(img_path) as img:
                # Convert RGBA to RGB (in case of PNG)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new aspect ratio
                width, height = img.size
                if max(width, height) > max_size:
                    if width > height:
                        new_width = max_size
                        new_height = int(height * (max_size / width))
                    else:
                        new_height = max_size
                        new_width = int(width * (max_size / height))
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Compress and save
                img.save(out_path, "JPEG", quality=quality)
                
            if (idx + 1) % 100 == 0 or (idx + 1) == len(files):
                print(f"Processed {idx + 1}/{len(files)} images.")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    raw_dir = r"c:\Users\B2B\Desktop\miniproject\dataset\raw_images"
    compressed_dir = r"c:\Users\B2B\Desktop\miniproject\dataset\compressed_images"
    compress_images(raw_dir, compressed_dir)
```
