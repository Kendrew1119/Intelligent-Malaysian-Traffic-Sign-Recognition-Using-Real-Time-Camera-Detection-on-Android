# MYSignVoice — Storage & Dataset Extraction Guide

If you have downloaded large `.rar` or `.zip` files (several gigabytes) and are running out of local computer storage, follow this step-by-step guide to extract, compress, and clean up the dataset safely.

---

## 🧭 Choose Your Pipeline

Depending on how much free disk space you have left on your computer, choose **Option A** or **Option B**:

* **Option A: Local Processing** — Use this if you have **at least 15 GB** of free space on your local hard drive.
* **Option B: Cloud Processing (Google Colab)** — Use this if you have **less than 5 GB** of free space. This option uses Google Colab's cloud storage (~100 GB) and does not consume your local disk space.

---

## 💻 Option A: Local Processing (Free Space > 15 GB)

Follow these steps on your local computer:

### Step 1: Create a Temporary Extraction Folder
Do not extract the files directly into your project folder. Create a temporary folder on your disk where you have the most space (e.g., `C:\temp_dataset` or `D:\temp_dataset`).

### Step 2: Extract the `.rar` Files
1. Install a free extractor like **7-Zip** (recommended) or WinRAR.
2. Right-click your `.rar` file → select **7-Zip** → **Extract to "C:\temp_dataset"**.
3. Once extracted, check the folder structure to find where the images are stored.

### Step 3: Run the Image Compression Script
We will use a Python script to scan your temporary folder, resize all images to **800px**, compress them, and save them directly to your project's local folder: `c:\Users\B2B\Desktop\miniproject\dataset\raw_images\`.

1. Open your terminal/command prompt.
2. Install the image processing library (Pillow):
   ```bash
   pip install Pillow
   ```
3. Run the compression script below.

### Step 4: Storage Cleanup (Crucial)
After the script finishes successfully and you verify that your compressed images are inside `dataset/raw_images/`:
1. Delete the temporary folder `C:\temp_dataset` entirely.
2. Delete the downloaded `.rar` file.
3. Empty your computer's **Recycle Bin** to permanently free up the disk space.

---

## ☁️ Option B: Cloud Processing (Free Space < 5 GB)

This pipeline uploads the large file to Google Drive and uses Google Colab to process it.

### Step 1: Upload the `.rar` File to Google Drive
Upload your large `.rar` file directly to your Google Drive (e.g., in a folder named `TrafficSignProject`).

### Step 2: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com) and start a new notebook.
2. Mount your Google Drive by running this code cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### Step 3: Extract the `.rar` File on the Cloud
Colab virtual machines have ~100 GB of free local storage. We will copy and extract the `.rar` file there:
1. Install `patool` (a library to extract RAR files):
   ```python
   !pip install patool
   ```
2. Extract the file to the Colab local drive:
   ```python
   import patoolib
   # Replace with your actual Google Drive folder and rar filename
   rar_path = "/content/drive/MyDrive/TrafficSignProject/dataset.rar"
   patoolib.extract_archive(rar_path, outdir="/content/extracted_raw")
   ```

### Step 4: Compress and Resize Images on Colab
Run this Python code cell in Colab to resize all extracted images to **800px** and compress them:

```python
import os
from PIL import Image

input_dir = "/content/extracted_raw"
output_dir = "/content/compressed_dataset"
os.makedirs(output_dir, exist_ok=True)

max_size = 800
quality = 85
supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Recursively find and process all images
count = 0
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(supported_extensions):
            img_path = os.path.join(root, file)
            out_path = os.path.join(output_dir, f"sign_{count}.jpg")
            try:
                with Image.open(img_path) as img:
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    width, height = img.size
                    if max(width, height) > max_size:
                        ratio = max_size / max(width, height)
                        img = img.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
                    
                    img.save(out_path, "JPEG", quality=quality)
                    count += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")

print(f"Successfully compressed {count} images into {output_dir}")
```

### Step 5: Zip and Save Back to Google Drive
1. Zip the compressed folder in Colab:
   ```python
   !zip -r -q /content/compressed_dataset.zip /content/compressed_dataset
   ```
2. Copy the zipped file back to your Google Drive:
   ```python
   !cp /content/compressed_dataset.zip /content/drive/MyDrive/TrafficSignProject/
   ```

### Step 6: Cleanup Google Drive & Download
1. Go to your Google Drive and **delete the original large `.rar` file** (and empty the Google Drive trash folder) to free up your Google Drive storage.
2. Download the small `compressed_dataset.zip` (which will only be ~150MB to 300MB instead of several gigabytes) to your local computer.
3. Extract this small file directly into your local project directory at `c:\Users\B2B\Desktop\miniproject\dataset\raw_images\`.
