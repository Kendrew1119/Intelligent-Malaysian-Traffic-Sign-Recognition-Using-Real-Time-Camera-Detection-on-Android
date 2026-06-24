# UCCC2513 Mini Project — Full 14-Week Plan

# 🦯 Intelligent Malaysian Traffic Sign Recognition for Visually Impaired Pedestrians Using Real-Time Camera Detection on Android

---

## 📋 Project Overview

**Course**: UCCC2513 Mini Project
**Team Size**: 4 members
**Duration**: 14 weeks
**Skill Profile**: C++ strong, Android & deep learning are new (plan is written for rookies in Android/ML)
**Assessments**:
- Assignment (Proposal Report + Presentation): **30 marks** — due ~Week 7
- Class Participation + Activities: **10 marks** — ongoing
- Project Work (Final Report + Presentation): **50 marks** — due ~Week 11–13

**Testing Environment**:
- **Primary**: Android phone (real device testing with camera)
- **Demo/Review on PC**: Use Android Studio Emulator with webcam passthrough, or screen-mirror the phone to PC via `scrcpy` (free tool) for presentation demos

**GPU for Training**:
- **No local GPU needed** — use **Google Colab** (free tier provides NVIDIA T4 GPU). Training YOLOv8-nano on Colab takes ~30–60 min for 100 epochs on a small dataset. See the [Google Colab Training Guide](#-google-colab-training-guide) section below.

---

## 🏷️ Suggested Project Titles

> The title must clearly state: **traffic sign detection** + **autonomous vehicles / assisted mobility** (as per assignment brief).

| # | Title | Why It Works |
|---|-------|-------------|
| 1 | **Intelligent Malaysian Traffic Sign Recognition for Visually Impaired Pedestrians Using Real-Time Camera Detection on Android** | Unique angle — accessibility for blind users, innovative, strong social impact, justifies camera + mobile |
| 2 | **Real-Time Malaysian Traffic Sign Detection and Audio Feedback System for Visually Impaired Users Using Edge AI on Android** | Emphasizes real-time + audio + edge AI |
| 3 | **MYSignVoice: A Camera-Based Malaysian Road Sign Detection and Text-to-Speech System for Visually Impaired Pedestrians** | Branded name (MYSignVoice), clearly describes what the app does |

> **Recommendation**: Title **1** or **3**. Title 1 is more formal/academic. Title 3 has a catchy app name.

---

## 💡 The App — Blind-Friendly Road Sign Audio Assistant

### Concept

An Android app designed for **visually impaired pedestrians** that uses the phone camera to continuously scan the environment, detect Malaysian road signs, and **speak aloud** the sign's meaning using Text-to-Speech (TTS). The app runs with the phone held in hand or worn around the neck using a lanyard.

### Why This Idea Wins the "Funding Competition"

The assignment frames the proposal as competing for RM 1,000,000 funding. This idea wins because:
- **Social impact** — improves road safety for 250,000+ visually impaired Malaysians (Department of Social Welfare statistics)
- **Clear innovation** — no existing Malaysian road sign app targets the visually impaired
- **Naturally justifies every technical choice**:
  - **Why camera?** — Blind users can't upload images or browse a gallery
  - **Why mobile app not web?** — Must be portable, works outdoors, hands-free
  - **Why real-time?** — Blind users need instant audio feedback, not batch processing
  - **Why Malaysian signs?** — Localized accessibility tool with genuine social need
  - **Why C++ + YOLO?** — Performance-critical real-time inference on mobile

### Complete Feature List

| Feature | Description | Technical Component |
|---------|-------------|-------------------|
| **Real-time camera detection** | Continuous scanning at 15–30 FPS, no tap needed | CameraX + YOLOv8n + ncnn |
| **Text-to-Speech output** | Speaks sign name + meaning: *"Warning: sharp bend ahead"* | Android TTS API |
| **Vibration alerts** | Phone vibrates for danger/warning signs (red triangles, stop signs) | Android Vibrator API |
| **Multilingual TTS** | Supports Bahasa Melayu and English, switchable in settings | Android TTS language packs |
| **High-contrast accessible UI** | Extra-large buttons, dark background, minimal clutter, designed for low/no vision | Jetpack Compose with accessibility guidelines |
| **Voice commands** | "Start" / "Stop" / "Change language" via speech recognition | Android SpeechRecognizer API |
| **Detection suppression** | Won't repeat the same sign if user is stationary (cooldown timer per sign class) | Custom logic: track last detected class + timestamp |
| **Distance estimation** | Audio cue: "near" / "medium" / "far" based on bounding box size relative to frame | Bounding box area ratio calculation |
| **Sign meaning database** | JSON database mapping 50+ sign IDs → name → full description in BM + EN | SQLite or JSON asset file |
| **Detection history log** | Saves last 50 detections with timestamp, sign name, confidence score | Room database (Android) |
| **Confidence threshold slider** | Adjustable sensitivity (default 0.5, range 0.3–0.9) | Settings screen |
| **PC demo mode** | Screen mirror via scrcpy for presentation; also works on Android Studio emulator | scrcpy / emulator with webcam |

### App Screen Flow

```
[Splash Screen] → [Main Camera Screen] → [Settings]
                         ↓                      ↓
                  [Detection Overlay]    [Language Toggle]
                  [TTS Speaking]         [Confidence Slider]
                  [Vibration Alert]      [Voice Command Toggle]
                         ↓
                  [History Screen]
```

### Malaysian Road Signs to Support (Minimum 50 Classes)

| Category | Color | Examples | Count |
|----------|-------|----------|-------|
| **Prohibitory** | Red border, white center | Speed limit (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110), No entry, No U-turn, No overtaking, No parking, No stopping | ~20 |
| **Warning** | Yellow/amber, black symbol | Sharp bend, curve, junction, roundabout, speed bump, pedestrian crossing, school zone, animal crossing, slippery road | ~15 |
| **Mandatory** | Blue, white symbol | Turn left, turn right, go straight, roundabout direction, one-way, keep left | ~10 |
| **Information** | Blue rectangle | Highway signs, directional signs, facility signs | ~5+ |
| **Stop / Special** | Red octagon / others | Stop sign, give way (inverted triangle) | ~3 |

---

## 🛠️ Tech Stack

### Core Development

| Component | Technology | Why | Beginner Tip |
|-----------|-----------|-----|-------------|
| **Core Logic** | **C++17** | Required by course. Used for image processing + ML inference | You already know this well |
| **Android App** | **Kotlin** | Official Android language, interops with C++ via JNI | Similar to Java but more concise |
| **Camera** | **CameraX (Android Jetpack)** | Google's modern camera API. Handles rotation, lifecycle, focus automatically | Much easier than old Camera2 API |
| **UI Framework** | **Jetpack Compose** | Declarative UI — describe what you want, not how to build it | Like writing HTML but in Kotlin |
| **Build System** | **CMake + Gradle** | CMake for C++ native lib, Gradle for Android app | Android Studio handles most of this |

### Camera Compatibility

CameraX works on **any Android phone running Android 5.0 (API 21) or above** — this covers 99%+ of all Android phones in use. It automatically handles:
- Front/back camera switching
- Auto-focus and auto-exposure
- Image rotation correction
- Frame capture for ML inference (via `ImageAnalysis` use case)

For this project, we use the **back camera** in `ImageAnalysis` mode, which delivers frames as `ImageProxy` objects that we convert to `Bitmap` → `Mat` (OpenCV) → feed to YOLO.

---

## 🤖 Why YOLOv8-nano — Deep Comparison

### YOLO Version Comparison

| Version | Year | Architecture | Speed (ms) on Mobile | mAP (COCO) | Model Size | Good for Mobile? |
|---------|------|-------------|---------------------|------------|------------|-----------------|
| YOLOv5-nano | 2021 | CSPDarknet + PANet | ~25ms | 28.0% | 3.9 MB | ✅ Yes |
| YOLOv7-tiny | 2022 | E-ELAN | ~30ms | 38.7% | 12.1 MB | ⚠️ OK but larger |
| **YOLOv8-nano** | **2023** | **C2f + SPPF + Decoupled Head** | **~20ms** | **37.3%** | **6.2 MB** | **✅ Best choice** |
| YOLOv8-small | 2023 | Same but wider | ~35ms | 44.9% | 22.5 MB | ⚠️ Too big for some phones |
| YOLOv8-medium | 2023 | Same but deeper | ~80ms | 50.2% | 52.0 MB | ❌ Too slow for real-time |
| YOLOv9-tiny | 2024 | GELAN + PGI | ~22ms | 38.3% | 7.7 MB | ✅ Yes but newer, less community support |
| YOLOv10-nano | 2024 | NMS-free design | ~18ms | 38.5% | 5.4 MB | ✅ Yes but very new, ncnn support may be limited |

### YOLOv8 vs YOLOv8-nano — What's the Difference?

YOLOv8 is a **family** of models, not a single model. The "nano" is the **smallest variant**:

```
YOLOv8 Family (from smallest to largest):
┌─────────────┬────────┬──────────┬───────────┬─────────────┐
│   Variant   │ Params │   Size   │ Speed(ms) │    Use For  │
├─────────────┼────────┼──────────┼───────────┼─────────────┤
│ YOLOv8n     │  3.2M  │  6.2 MB  │   ~20ms   │ 📱 Mobile   │  ← WE USE THIS
│ YOLOv8s     │  11.2M │  22.5 MB │   ~35ms   │ 💻 Edge     │
│ YOLOv8m     │  25.9M │  52.0 MB │   ~80ms   │ 🖥️ Desktop  │
│ YOLOv8l     │  43.7M │  87.7 MB │  ~120ms   │ 🖥️ Server   │
│ YOLOv8x     │  68.2M │ 136.7 MB │  ~160ms   │ 🏢 Cloud    │
└─────────────┴────────┴──────────┴───────────┴─────────────┘
```

**Key differences**:
- **n (nano)**: Fewer convolutional layers, narrower channels (e.g., 64 → 128 → 256 instead of 256 → 512 → 1024). Fastest but lowest accuracy.
- **s/m/l/x**: Progressively deeper and wider networks. More accurate but slower.
- All share the **same architecture design** (C2f blocks, SPPF, decoupled detection head), just scaled differently.

### Why YOLOv8-nano is Perfect for This Project

1. **6.2 MB model file** — fits easily in an Android APK (the whole app stays under 50 MB)
2. **~20ms inference** on a mid-range phone — gives us 50 FPS, far exceeding the 2-second requirement
3. **37.3% mAP on COCO** — for our focused task (just road signs, not 80 COCO classes), fine-tuned accuracy will be much higher (typically 85–95% mAP)
4. **Excellent ncnn support** — Tencent maintains official YOLOv8 examples for ncnn
5. **Ultralytics ecosystem** — easiest training framework: one command to train, one command to export
6. **Huge community** — thousands of tutorials, Stack Overflow answers, GitHub issues

### What Camera Resolution Does YOLOv8-nano Expect?

YOLOv8-nano takes **640×640 pixel** input by default. Here's how the pipeline works:

```
Phone Camera (1920×1080) → Resize to 640×640 → YOLOv8-nano → Bounding boxes → Scale back to 1920×1080 for display
```

You **don't** need a 640×640 camera. CameraX captures at whatever resolution is available (usually 1080p or 720p), and our code resizes the frame before feeding it to YOLO. The resize is handled by OpenCV's `cv::resize()` in C++.

---

## 🔗 ncnn Integration — Step-by-Step Guide for Beginners

This is the most complex part of the project. Follow these steps carefully.

### What is ncnn?

ncnn is a **C++ neural network inference framework** by Tencent. Think of it as: "a C++ library that can load a trained AI model and run predictions." It's:
- Written in pure C++ (no Python, no Java dependency)
- Designed for mobile (Android/iOS)
- Very fast on ARM processors
- No need for GPU on the phone — runs on CPU efficiently

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Android App (Kotlin)                │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  CameraX    │  │     UI       │  │    TTS     │ │
│  │  (frames)   │  │  (display)   │  │  (speech)  │ │
│  └──────┬──────┘  └──────▲───────┘  └─────▲──────┘ │
│         │                │                │         │
│         ▼                │                │         │
│  ┌──────────────────────────────────────────────┐   │
│  │              JNI Bridge (Kotlin ↔ C++)       │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                           │
│  ┌──────────────────────▼───────────────────────┐   │
│  │              C++ Native Library               │   │
│  │                                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────┐ │   │
│  │  │  OpenCV   │  │   ncnn   │  │   Post-    │ │   │
│  │  │ (resize,  │  │  (YOLO   │  │ processing │ │   │
│  │  │  convert) │  │ inference│  │  (NMS,     │ │   │
│  │  │           │  │          │  │  labels)   │ │   │
│  │  └──────────┘  └──────────┘  └────────────┘ │   │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Step-by-Step Integration

#### Step 1: Train YOLOv8-nano (Python, on Google Colab)

```python
# In Google Colab
!pip install ultralytics

from ultralytics import YOLO

# Load pretrained YOLOv8-nano
model = YOLO('yolov8n.pt')

# Train on your Malaysian road sign dataset
model.train(
    data='road_signs.yaml',  # Your dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    name='malaysian_signs'
)

# Export to ONNX format
model.export(format='onnx', imgsz=640, simplify=True)
```

#### Step 2: Convert ONNX to ncnn format (on PC)

```bash
# Install ncnn tools (or use prebuilt binaries from ncnn releases)
# Download from: https://github.com/Tencent/ncnn/releases

# Convert ONNX to ncnn
./onnx2ncnn best.onnx yolov8n_signs.param yolov8n_signs.bin

# Optimize the model (reduces size, speeds up inference)
./ncnnoptimize yolov8n_signs.param yolov8n_signs.bin yolov8n_signs_opt.param yolov8n_signs_opt.bin 65536
```

This produces two files:
- `yolov8n_signs_opt.param` — model architecture (text file, ~50 KB)
- `yolov8n_signs_opt.bin` — model weights (binary file, ~6 MB)

#### Step 3: Add ncnn to Android project

In your Android project's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.22)
project(signdetector)

# Find ncnn package (prebuilt for Android)
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-android-lib/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# Find OpenCV
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

# Your C++ source files
add_library(signdetector SHARED
    src/main/cpp/sign_detector.cpp
    src/main/cpp/jni_bridge.cpp
)

# Link libraries
target_link_libraries(signdetector ncnn ${OpenCV_LIBS} android log)
```

#### Step 4: Write C++ inference code

```cpp
// sign_detector.cpp — simplified example
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>

class SignDetector {
public:
    ncnn::Net net;
    
    bool loadModel(const char* paramPath, const char* binPath) {
        net.opt.use_vulkan_compute = false;  // CPU only, simpler
        net.opt.num_threads = 4;             // Use 4 CPU cores
        net.load_param(paramPath);
        net.load_model(binPath);
        return true;
    }
    
    std::vector<Detection> detect(cv::Mat& frame) {
        // 1. Convert BGR to RGB and resize to 640x640
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 640));
        
        // 2. Create ncnn input (normalize pixels to 0-1)
        ncnn::Mat input = ncnn::Mat::from_pixels(
            resized.data, ncnn::Mat::PIXEL_RGB, 640, 640);
        const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
        const float mean[3] = {0.f, 0.f, 0.f};
        input.substract_mean_normalize(mean, norm);
        
        // 3. Run inference
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", input);
        ncnn::Mat output;
        ex.extract("out0", output);
        
        // 4. Parse output into detections
        // (parse bounding boxes, class IDs, confidence scores)
        // ... post-processing code here ...
        
        return detections;
    }
};
```

#### Step 5: Write JNI bridge

```cpp
// jni_bridge.cpp
#include <jni.h>
#include "sign_detector.h"

static SignDetector* detector = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_myapp_signdetector_NativeLib_initModel(
    JNIEnv* env, jobject, jstring paramPath, jstring binPath) {
    
    detector = new SignDetector();
    const char* param = env->GetStringUTFChars(paramPath, nullptr);
    const char* bin = env->GetStringUTFChars(binPath, nullptr);
    bool ok = detector->loadModel(param, bin);
    env->ReleaseStringUTFChars(paramPath, param);
    env->ReleaseStringUTFChars(binPath, bin);
    return ok;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_myapp_signdetector_NativeLib_detect(
    JNIEnv* env, jobject, jlong framePtr) {
    
    cv::Mat* frame = reinterpret_cast<cv::Mat*>(framePtr);
    auto results = detector->detect(*frame);
    // Convert results to Java array and return
    // ...
}
```

#### Step 6: Call from Kotlin

```kotlin
// NativeLib.kt
class NativeLib {
    companion object {
        init { System.loadLibrary("signdetector") }
    }
    external fun initModel(paramPath: String, binPath: String): Boolean
    external fun detect(framePtr: Long): Array<Detection>
}
```

### Where to Get Pre-built ncnn for Android

Download the latest Android prebuilt from:
**https://github.com/Tencent/ncnn/releases**

Look for: `ncnn-YYYYMMDD-android-vulkan.zip`

Extract and place in your project's `app/src/main/cpp/ncnn-android-lib/` directory.

### Fallback: If ncnn is Too Complex

If the team struggles with ncnn integration, use **TensorFlow Lite (TFLite)** instead:
- Also supports C++ API
- Slightly slower than ncnn but has more tutorials
- Export: `model.export(format='tflite')` — no ONNX step needed
- Android integration is well-documented by Google

---

## ☁️ Google Colab Training Guide

Since the team has no local GPU, use Google Colab (free).

### Step-by-Step

1. Go to **https://colab.research.google.com**
2. Create a new notebook
3. **Change runtime**: `Runtime` → `Change runtime type` → select `T4 GPU`
4. Upload your dataset (or mount Google Drive)
5. Run:

```python
# Cell 1: Install
!pip install ultralytics

# Cell 2: Mount Google Drive (to save model)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Upload dataset to Colab or use from Drive
# Your dataset structure should be:
# /content/dataset/
#   ├── train/
#   │   ├── images/
#   │   └── labels/
#   ├── val/
#   │   ├── images/
#   │   └── labels/
#   └── data.yaml

# Cell 4: Train
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # nano variant
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,       # Early stopping if no improvement for 20 epochs
    project='/content/drive/MyDrive/sign_model',
    name='run1'
)

# Cell 5: Export to ONNX
model = YOLO('/content/drive/MyDrive/sign_model/run1/weights/best.pt')
model.export(format='onnx', imgsz=640, simplify=True)

# Model saved at: /content/drive/MyDrive/sign_model/run1/weights/best.onnx
```

### Training Time Estimate (Google Colab T4 GPU)

| Dataset Size | Epochs | Approx Time |
|-------------|--------|-------------|
| 200 images | 100 | ~15 min |
| 500 images | 100 | ~30 min |
| 1000 images | 100 | ~60 min |
| 2000 images | 150 | ~2 hours |

### Important Colab Tips

- Free tier has a **~4 hour** session limit — save checkpoints to Google Drive
- Training auto-saves checkpoints every epoch, so if disconnected you can resume
- Use `patience=20` for early stopping to avoid wasting time if model converges early

---

## 📅 14-Week Schedule

> Aligned with the **Teaching Plan** milestones (Week 2: title submission, Week 5: lit review draft, Week 7: proposal submission + presentation, Week 11–13: final presentation).

### Phase 1: Research & Proposal (Weeks 1–7)

| Week | Focus | Deliverables | Who |
|------|-------|-------------|-----|
| **1** | **Project Kickoff** | • Form team, assign roles (see Task Distribution below) • Read all course documents carefully • Read this entire plan.md • Set up GitHub repo + WhatsApp/Discord group • Each member installs Android Studio | All |
| **2** | **Title & Literature Search** | • Submit project title + team list to lecturer • Each member searches 5–8 papers on their assigned keywords (see Lit Review section) • Start shared Google Doc for references | All |
| **3** | **Deep Literature Review** | • Each member writes 3-page critique for their area with proper citations • Draft Chapter 2 (Literature Review) in shared Google Doc • Group meeting to discuss findings and agree on system design | All |
| **4** | **Setup Dev Environment** | • Install & configure: Android Studio, CMake, OpenCV C++, Python, Ultralytics • Run a "Hello World" Android app with CameraX (Member 1 & 2) • Run a YOLOv8n inference test on a sample image in Python (Member 3 & 4) • Start photographing/downloading Malaysian road sign images | Member 1 & 2 (Android), Member 3 & 4 (ML/Python) |
| **5** | **Draft Proposal Chapters 1–3** | • Chapter 1 (Introduction) — *Member 1* • Chapter 2 (Lit Review) — combine all 4 reviews — *All* • Chapter 3 (Proposed Method) — system block diagram + flowchart — *Member 2* • Submit draft of Chapter 2 to supervisor | All |
| **6** | **Preliminary Work (Ch 4) + Finalize Proposal** | • Each member implements their preliminary module in C++ (see Ch 4 modules below) • Chapter 4 screenshots and results • Chapter 5 (Conclusion) — *Member 3* • Proofread, Turnitin check, format per FYP1 template | All |
| **7** | **🎯 PROPOSAL SUBMISSION + PRESENTATION** | • Submit proposal report (Ch 1–5) • Record/present presentation video with demo • Each member shows their segmentation results on test images | All |

### Phase 2: Development (Weeks 8–10)

| Week | Focus | Deliverables | Who |
|------|-------|-------------|-----|
| **8** | **Dataset + Model Training** | • Complete dataset: 300+ annotated Malaysian road sign images • Augment to 1500+ images • Train YOLOv8n on Google Colab • Evaluate mAP, iterate if needed • Export to ONNX → ncnn format | Member 3 (dataset) + Member 4 (training), Member 1 & 2 (help annotate + start Android scaffold) |
| **9** | **Android App Core** | • CameraX live preview + frame capture (Member 1) • ncnn C++ inference engine + JNI bridge (Member 2) • Sign database JSON + TTS mapping (Member 3) • Model optimization + testing pipeline (Member 4) • **Integration day**: combine camera → inference → display | Member 1 (camera+UI), Member 2 (C++/JNI), Member 3 (database), Member 4 (model) |
| **10** | **Features + Testing** | • TTS audio output working • Vibration alerts for warning signs • Test on all 84 provided images → log recognition rate • Test on real roads (phone in hand, walk around campus) • Bug fixes, performance tuning (<2 sec requirement) • PC demo setup via scrcpy | All |

### Phase 3: Final Report & Presentation (Weeks 11–14)

| Week | Focus | Deliverables | Who |
|------|-------|-------------|-----|
| **11** | **Final Report Writing** | • Follow FYP2 report template • Expand Ch 1–3 from proposal • Ch 4 (Implementation): each member documents their module • Ch 5 (Results): recognition rates, confusion matrix, error analysis | All |
| **12** | **Report Completion** | • Ch 6 (Conclusion + Future Work) — *Member 4* • Appendices: source code • Tag every section with member name • Proofread, Turnitin check • Report ≤ 60 pages | All |
| **13** | **🎯 FINAL PRESENTATION + DEMO** | • Record/present final video • Live demo: run app on phone, detect signs from camera • Show results for all 84 images + extra images (bonus 5 marks) • Screen mirror to PC for presentation via scrcpy | All |
| **14** | **Buffer / Polish** | • Address feedback • Final cleanup and submission | All |

---

## 👥 Task Distribution (4 Members)

### Role Assignments

| Member | Role Title | Primary Responsibility | Skills to Learn |
|--------|-----------|----------------------|----------------|
| **Member 1** | **Android & UI Lead** | CameraX, Jetpack Compose UI, TTS, vibration, accessibility | Kotlin basics, CameraX tutorial, Jetpack Compose |
| **Member 2** | **Systems Architect** | C++ ncnn engine, JNI bridge, OpenCV integration, system design | JNI tutorial, ncnn examples, CMake for Android |
| **Member 3** | **Data & Database Lead** | Dataset collection, annotation, augmentation, sign meaning database | Roboflow, albumentations, SQLite/JSON |
| **Member 4** | **ML & Evaluation Lead** | YOLO training (Colab), model conversion, testing, performance analysis | Ultralytics docs, ONNX export, Google Colab |

---

### Report Task Distribution

#### Proposal Report (30 marks, due Week 7)

| Chapter | Content | Assigned To | Marks |
|---------|---------|-------------|-------|
| **Ch 1 — Introduction** | Background, problem statement, objectives, scope, significance | **Member 1** | 3 |
| **Ch 2 — Literature Review** | Each member writes their subtopic (see below) | **All (3 pages each)** | 12 |
| **Ch 3 — Proposed Method** | System block diagram, flowchart, module descriptions, W5H | **Member 2** | 8 |
| **Ch 4 — Preliminary Work** | Each member's color/shape segmentation results | **Each separately** | 5 |
| **Ch 5 — Conclusion** | Summary, expected outcomes, timeline | **Member 3** | 2 |
| **Presentation** | 10-min video: system overview + demo | **Member 4 (coordinator), All present** | 10 |

**Chapter 4 — Preliminary Work Modules** (required by assignment):

| Member | Module | Description |
|--------|--------|-------------|
| Member 1 | Red sign segmentation using color | Convert to HSV, threshold for red hue range (H: 0-10 & 170-180, S: 70-255, V: 50-255), morphological operations, extract red sign regions |
| Member 2 | Blue sign segmentation using color | HSV threshold for blue hue range (H: 100-130, S: 50-255, V: 50-255), extract blue sign regions |
| Member 3 | Yellow sign segmentation using color | HSV threshold for yellow hue range (H: 15-35, S: 80-255, V: 80-255), extract yellow sign regions |
| Member 4 | Shape detection of signs | Canny edge detection → findContours → approxPolyDP to classify: circle (>8 vertices), triangle (3), rectangle (4), octagon (8) |

> ⚠️ Each member's code and writing sections **must be tagged with their name** — "No name no mark" policy.

---

#### Final Report (50 marks, due Week 11–13)

| Chapter | Content | Assigned To |
|---------|---------|-------------|
| **Ch 1 — Introduction** | Updated from proposal, refined objectives | **Member 1** |
| **Ch 2 — Literature Review** | Expanded with development findings | **All** |
| **Ch 3 — Methodology** | YOLO architecture, ncnn pipeline, Android architecture, TTS design | **Member 2** |
| **Ch 4 — Implementation** | Code walkthrough per module | **Each member** |
| **Ch 5 — Results & Analysis** | Recognition rates, confusion matrix, error analysis, failure conditions | **Member 3 + Member 4** |
| **Ch 6 — Conclusion & Future Work** | Summary, limitations, enhancements | **Member 4** |
| **Appendices** | Source code listings | **All** |
| **Presentation + Demo** | Final video + live demo | **Member 1 (demo), All** |

**Individual Module Tasks** (required by project work doc):

| Task | Member | Description |
|------|--------|-------------|
| **Task 1: Features (A)** | **Member 3** | HOG descriptors, color histograms, edge features → report recognition rates per feature set |
| **Task 1: Features (B)** | **Member 4** | LBP texture, shape descriptors, aspect ratio features → report recognition rates independently |
| **Task 2: Classifiers (A)** | **Member 1** | SVM and KNN classifiers → report recognition rates |
| **Task 2: Classifiers (B)** | **Member 2** | Random Forest and CNN/YOLO → report recognition rates |

---

### Development Task Distribution

| Component | Member | Details |
|-----------|--------|---------|
| **CameraX + Live Preview** | Member 1 | Camera feed, frame capture at 15-30 FPS, lifecycle management |
| **Accessible UI Design** | Member 1 | High-contrast dark theme, extra-large buttons, screen reader support |
| **TTS / Audio Output** | Member 1 | Sign ID → spoken description mapping, BM/EN language toggle |
| **Vibration Alerts** | Member 1 | Vibrate patterns for different sign severity levels |
| **C++ ncnn Inference Engine** | Member 2 | Load model, run YOLO inference, NMS post-processing |
| **JNI Bridge** | Member 2 | Kotlin ↔ C++ interop, pass camera frames, return detections |
| **OpenCV Pre/Post-processing** | Member 2 | Resize, color convert, draw bounding boxes |
| **Dataset Collection** | Member 3 | Photograph + download 300+ Malaysian road signs |
| **Data Annotation** | Member 3 | Roboflow bounding box annotation for all images |
| **Data Augmentation** | Member 3 | albumentations: flip, rotate, brightness, noise |
| **Sign Database** | Member 3 | JSON file: sign_id → name_en, name_bm, description_en, description_bm, severity |
| **YOLO Training (Colab)** | Member 4 | Train YOLOv8n, tune hyperparameters, validate mAP |
| **Model Export & Conversion** | Member 4 | PyTorch → ONNX → ncnn, test converted model |
| **Performance Testing** | Member 4 | Test 84 images, measure accuracy + speed, confusion matrix |
| **Detection Suppression Logic** | Member 4 | Cooldown timer to avoid repeating same sign announcement |

---

## 📚 Literature Review — Detailed Keywords & Guide

### How to Search for Papers

1. Use **Google Scholar** (scholar.google.com)
2. Filter by year: **2019–2026** (last 5–7 years)
3. Look for papers with **50+ citations** (reliable)
4. Prefer: IEEE, Springer, Elsevier, ACM, MDPI journals/conferences
5. Each member finds **5–8 papers**, picks best **3–5** to review in depth

---

### Member 1: Color-Based Road Sign Segmentation

**Search Keywords** (combine 2–3 for each search):

| Primary Keywords | Combine With |
|-----------------|-------------|
| `color segmentation traffic sign` | `HSV`, `color space`, `thresholding` |
| `red sign detection color` | `autonomous driving`, `real-time` |
| `color-based road sign detection` | `HSV threshold`, `LAB color space` |
| `traffic sign segmentation colour` | `morphological operations`, `connected components` |
| `colour filtering traffic signs` | `adaptive thresholding`, `Otsu` |

**Key Concepts to Cover in Review**:
- HSV vs RGB vs LAB color spaces — which is best for sign detection and why
- Adaptive vs fixed thresholding methods
- Morphological operations (erosion, dilation, opening, closing) for noise removal
- Challenges: varying lighting, shadows, weathered signs
- Compare 3–4 papers' approaches and their accuracy results

**Example Papers to Look For**:
- "Real-time traffic sign detection using color and shape information" (common title pattern)
- "A survey of traffic sign recognition" (survey papers are great for literature review)
- "Robust traffic sign detection in challenging lighting conditions"

---

### Member 2: Shape Detection & Geometric Analysis for Road Signs

**Search Keywords**:

| Primary Keywords | Combine With |
|-----------------|-------------|
| `shape detection traffic sign` | `Hough Transform`, `contour analysis` |
| `geometric feature road sign` | `circle detection`, `polygon recognition` |
| `traffic sign shape classification` | `edge detection`, `Canny` |
| `contour-based sign recognition` | `approxPolyDP`, `convex hull` |
| `road sign shape segmentation` | `template matching`, `moment invariants` |

**Key Concepts to Cover in Review**:
- Hough Circle Transform for circular sign detection
- Contour detection + polygon approximation (approxPolyDP)
- Hu Moments and shape descriptors
- Edge detection methods: Canny, Sobel, Laplacian
- Template matching approaches
- Compare accuracy of shape-only vs color+shape methods

**Example Papers to Look For**:
- "Traffic sign detection using Hough Transform and shape analysis"
- "Shape-based traffic sign recognition using contour features"
- "Multi-feature traffic sign detection combining color and shape"

---

### Member 3: Deep Learning for Traffic Sign Recognition (YOLO, CNN)

**Search Keywords**:

| Primary Keywords | Combine With |
|-----------------|-------------|
| `deep learning traffic sign recognition` | `YOLO`, `CNN`, `real-time` |
| `YOLOv8 traffic sign detection` | `mobile`, `edge computing` |
| `convolutional neural network road sign` | `transfer learning`, `fine-tuning` |
| `traffic sign detection deep learning` | `small object detection`, `data augmentation` |
| `GTSRB GTSDB benchmark` | `recognition accuracy`, `comparison` |
| `lightweight object detection mobile` | `YOLOv5`, `SSD`, `MobileNet` |

**Key Concepts to Cover in Review**:
- Evolution: traditional CV → CNN → YOLO/SSD for sign detection
- YOLO architecture overview (one-stage vs two-stage detectors)
- Transfer learning: pretrained model → fine-tune on sign dataset
- German Traffic Sign Recognition Benchmark (GTSRB) — the standard benchmark
- Data augmentation techniques for small datasets
- Comparison of YOLO vs SSD vs Faster R-CNN accuracy/speed
- Why lightweight models (nano/tiny) are needed for mobile deployment

**Example Papers to Look For**:
- "YOLOv8 for real-time traffic sign detection"
- "A comprehensive survey on traffic sign detection and recognition"
- "Lightweight traffic sign detection for embedded systems"
- "Transfer learning for traffic sign recognition with limited data"

---

### Member 4: Accessibility Technology + Mobile AI / Alternative Approaches

**Search Keywords**:

| Primary Keywords | Combine With |
|-----------------|-------------|
| `assistive technology visually impaired` | `object detection`, `mobile app` |
| `blind navigation mobile camera` | `real-time`, `text-to-speech` |
| `accessibility AI visual impairment` | `Android`, `deep learning` |
| `traffic sign recognition mobile deployment` | `TensorFlow Lite`, `ncnn`, `edge AI` |
| `alternative traffic sign detection` | `feature extraction`, `SVM`, `template matching` |
| `on-device machine learning Android` | `model optimization`, `quantization` |

**Key Concepts to Cover in Review**:
- Existing assistive technology apps for the visually impaired (e.g., Seeing AI by Microsoft, Be My Eyes)
- Mobile deployment of ML models: TFLite vs ncnn vs ONNX Runtime
- Model quantization (FP32 → FP16 → INT8) for speed
- Alternative approaches: HOG + SVM, template matching, feature-based methods
- Comparison of deep learning vs traditional CV methods for this specific task
- Human-computer interaction for visually impaired users
- Text-to-speech systems and accessibility design principles

**Example Papers to Look For**:
- "A review of assistive technology for the visually impaired"
- "On-device deep learning inference on mobile devices: a survey"
- "Traffic sign recognition using HOG features and SVM classifier"
- "Comparison of traditional and deep learning approaches for traffic sign detection"

---

### Literature Review Writing Structure (for each member)

Each member's 3-page section should follow this structure:

```
2.X [Your Subtopic Title]    (tagged: written by [Your Name])

Paragraph 1: Overview of the technique category
  - What is it? General introduction to the approach
  - Why is it relevant to traffic sign detection?

Paragraph 2-3: Paper-by-paper review
  - For each paper: Author (Year) proposed [method]. They achieved [result] on [dataset].
  - Strengths: what worked well
  - Weaknesses: limitations they mentioned or you identified
  - Compare with other papers in your section

Paragraph 4: Summary and gap analysis
  - What do existing approaches do well?
  - What gaps exist? (this is where our proposed method fills in)
  - How does this inform our system design?
```

---

## ✅ Marking Scheme Alignment Checklist

### Proposal (30 marks)

| Criteria | How We Address It |
|----------|------------------|
| Ch 1: Clear problem statement + objectives (3 marks) | Accessibility for visually impaired + traffic sign detection |
| Ch 2: Recent literature within 5–10 years (12 marks) | Each member reviews 3–5 papers from 2019–2026 |
| Ch 3: Novel system design with block diagram (8 marks) | YOLO + ncnn + Android + TTS pipeline diagram |
| Ch 4: Working preliminary segmentation (5 marks) | 4 modules: Red/Blue/Yellow color segmentation + shape detection |
| Ch 5: Clear conclusion (2 marks) | Summary + expected outcomes + timeline |
| Presentation: Clear demo video (10 marks) | Show segmentation results on test images |

### Final Project (50 marks)

| Criteria | How We Address It |
|----------|------------------|
| Well-written report following FYP2 format (20 marks) | Follow template, system diagram, W5H explanation |
| Runs within 2 seconds per image (5 marks) | ncnn YOLO inference ~20ms per frame, well under 2s |
| Correct sign identification × 84 images (70 marks scaled) | Train YOLO on diverse Malaysian sign dataset |
| Beyond 84 test images (5 marks) | Test on our own collected Malaysian road sign photos |
| Report ≤ 60 pages (1 mark) | Monitor page count during writing |
| Each section tagged with member name | Enforce naming convention in every section |

---

## 🔧 Tools & Software Setup Summary

| Tool | Purpose | How to Get |
|------|---------|-----------|
| Android Studio Ladybug (2024+) | Android app development + Kotlin | developer.android.com/studio |
| Android NDK + CMake | Compile C++ for Android | Install via Android Studio → SDK Manager → SDK Tools |
| OpenCV 4.x Android SDK | Image processing in C++ | opencv.org/releases → Android |
| ncnn (prebuilt Android) | C++ neural network inference | github.com/Tencent/ncnn/releases |
| Python 3.10+ | Training scripts | python.org |
| Ultralytics | Train YOLOv8 | `pip install ultralytics` |
| Google Colab | Free GPU for training | colab.research.google.com |
| Roboflow | Dataset annotation (free tier) | roboflow.com |
| scrcpy | Mirror Android screen to PC for demos | github.com/Genymobile/scrcpy |
| Git + GitHub | Version control | github.com |
| MS Visual Studio 2022 | C++ dev/testing on PC (preliminary work) | visualstudio.microsoft.com |

---

## ⚠️ Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Not enough Malaysian sign images | Low accuracy | Start collecting Week 2, augment 5–10x, supplement with GTSDB dataset |
| YOLO training needs GPU | Can't train locally | Use Google Colab (free T4 GPU) — see guide above |
| ncnn integration is complex | Blocks development | Follow step-by-step guide above; fallback to TensorFlow Lite if stuck |
| Android/Kotlin is new to team | Slow development | Follow CameraX codelabs by Google; start simple, iterate |
| App too slow on phone | Fails 2-sec requirement | YOLOv8-nano ~20ms inference; quantize to INT8 if needed |
| Team member falls behind | Incomplete work | Weekly check-ins, clear ownership, buffer week 14 |
| Signs look different in real-world vs training | Poor generalization | Augment data heavily; test in various lighting conditions |

---

## 🤖 AI Agent Prompts for Team Members

> These prompts are for teammates to use with AI coding assistants (Gemini, Cursor, etc.) when building their assigned modules. **Read the full plan.md first** before using any prompt. Each prompt references files and architecture from this plan.

---

### Member 1 — Android & UI Lead

#### Prompt: CameraX Setup
```
I am building an Android app in Kotlin using Jetpack Compose. I need to set up CameraX 
to capture live camera frames for real-time ML inference.

Requirements:
- Use the BACK camera
- Set up ImageAnalysis use case to capture frames at 15-30 FPS
- Each frame should be converted from ImageProxy to android.graphics.Bitmap
- The Bitmap will be passed to a C++ native library via JNI for YOLOv8 inference
- Also set up a Preview use case so the user sees the live camera feed
- Handle camera permissions properly
- Use CameraX lifecycle binding

Tech stack: Kotlin, Jetpack Compose, CameraX (latest version), minimum API 24.
The app is a road sign detection app for visually impaired users.

Please create:
1. CameraScreen.kt - Composable with camera preview and overlay for detection boxes
2. CameraFrameAnalyzer.kt - ImageAnalysis.Analyzer that converts frames
3. Required permission handling code

Keep it beginner-friendly with comments explaining each step.
```

#### Prompt: Text-to-Speech Implementation
```
I need to implement Text-to-Speech (TTS) for an Android app (Kotlin, Jetpack Compose) 
that speaks detected road sign names aloud for visually impaired users.

Requirements:
- Initialize Android's TextToSpeech engine
- Support two languages: English (en-MY) and Bahasa Melayu (ms-MY)
- Language toggle in settings
- Function: speakSign(signId: Int, signName: String, description: String)
- Detection suppression: don't repeat the same sign within 5 seconds
- Queue management: if a new sign is detected while speaking, interrupt current speech
- Vibration alert for warning/danger signs (sign categories: prohibitory, warning, mandatory)
- Different vibration patterns: short buzz for info, long buzz for warning, double buzz for danger

Create:
1. TTSManager.kt - manages TTS engine lifecycle and speech
2. VibrationManager.kt - manages vibration patterns
3. SignDatabase.kt - JSON-based sign lookup (signId → name, description in EN and BM)

Include a sample JSON structure for the sign database.
```

#### Prompt: Accessible UI Design
```
I need to build an accessible Android UI using Jetpack Compose for a road sign detection 
app designed for visually impaired users.

Requirements:
- Dark background with high-contrast elements (WCAG AAA contrast ratio)
- Extra-large touch targets (minimum 72dp x 72dp)
- Only 3 main screens: Camera (main), History, Settings
- Camera screen: full-screen camera preview with translucent overlay showing detected sign name
- Big "Start/Stop Detection" toggle button at the bottom
- Settings: language toggle (EN/BM), confidence threshold slider, voice command toggle
- History screen: list of last 50 detected signs with timestamps
- Support Android TalkBack screen reader (proper content descriptions)
- Minimal text — the app relies on audio, not visual feedback
- Material 3 design with custom dark color scheme

Create the full UI scaffold with navigation.
```

---

### Member 2 — Systems Architect

#### Prompt: ncnn YOLOv8 Inference Engine (C++)
```
I need to write a C++ inference engine using ncnn to run a YOLOv8-nano model on Android.

Context:
- The model is trained with Ultralytics YOLOv8n on a custom dataset (Malaysian road signs)
- Model files: yolov8n_signs.param and yolov8n_signs.bin (ncnn format)
- Input: 640x640 RGB image
- Output: detected bounding boxes with class IDs and confidence scores
- Need NMS (Non-Maximum Suppression) post-processing with IoU threshold 0.45
- Confidence threshold default 0.5 (configurable)
- Number of classes: ~50 (Malaysian road sign types)

Requirements:
- Use ncnn C++ API (not the Java/Kotlin wrapper)
- Use OpenCV for image preprocessing (resize, color conversion)
- Implement proper NMS to filter overlapping detections
- Return results as a vector of structs: {x, y, width, height, classId, confidence}
- Use 4 CPU threads for inference
- No Vulkan GPU compute (keep it simple, CPU only)

Create:
1. sign_detector.h - header with Detection struct and SignDetector class
2. sign_detector.cpp - full implementation
3. Include detailed comments explaining each ncnn API call

I'm a C++ developer but new to ncnn. Please explain ncnn concepts as you go.
```

#### Prompt: JNI Bridge
```
I need to create a JNI (Java Native Interface) bridge to connect my Kotlin Android app 
with a C++ native library for YOLOv8 inference.

Context:
- C++ side: SignDetector class with loadModel() and detect() methods
- Kotlin side: NativeLib class that calls into C++
- Camera frames come as android.graphics.Bitmap from CameraX
- Need to pass Bitmap data to C++ as pixel array
- C++ processes with OpenCV + ncnn, returns detection results
- Results need to be passed back to Kotlin as a list of Detection objects

The C++ library uses OpenCV and ncnn. The Android app uses Kotlin + Jetpack Compose.

Create:
1. jni_bridge.cpp - C++ JNI functions
2. NativeLib.kt - Kotlin external function declarations
3. Detection.kt - Kotlin data class matching C++ Detection struct
4. CMakeLists.txt - complete CMake config linking ncnn + OpenCV + JNI

Include explanation of:
- How JNI works (for a beginner)
- How to pass Bitmap pixels from Kotlin to C++
- How to return array of objects from C++ to Kotlin
- Common JNI pitfalls and how to avoid memory leaks
```

---

### Member 3 — Data & Database Lead

#### Prompt: Dataset Preparation
```
I need to prepare a dataset of Malaysian road signs for YOLOv8 object detection training.

Current situation:
- I have 84 sample images in "Color Inputs" folder (Red Signs, Blue Signs, Yellow Signs)
- I need to collect more Malaysian road signs to reach 300+ images minimum
- Signs include: speed limits, warning signs, mandatory signs, prohibitory signs
- Need to annotate with bounding boxes and class labels

Please help me with:
1. A Python script to organize and rename images into a YOLO-compatible folder structure
2. A data.yaml configuration file for Ultralytics YOLOv8 training
3. A Python script using albumentations to augment the dataset (flip, rotate, brightness, 
   noise, blur, crop) to multiply it 5x
4. A list of recommended Malaysian road sign classes to use (at least 50 classes)
5. Instructions on how to use Roboflow (free tier) for bounding box annotation
6. A validation split script (80% train, 10% validation, 10% test)

YOLO annotation format: each image has a .txt file with lines of:
class_id center_x center_y width height (all normalized 0-1)
```

#### Prompt: Sign Meaning Database
```
I need to create a comprehensive JSON database of Malaysian road signs for a Text-to-Speech 
app that helps visually impaired users understand road signs detected by camera.

Requirements:
- Minimum 50 sign entries
- Each entry should have:
  - signId (int): matching the YOLO class ID
  - nameEn (string): sign name in English
  - nameBm (string): sign name in Bahasa Melayu
  - descriptionEn (string): what the sign means in simple English (spoken by TTS)
  - descriptionBm (string): what the sign means in Bahasa Melayu
  - category (string): "prohibitory", "warning", "mandatory", "information"
  - severity (string): "info", "caution", "danger"
  - shape (string): "circle", "triangle", "rectangle", "octagon", "diamond"
  - color (string): "red", "blue", "yellow", "green"

Categories of Malaysian road signs to include:
- Speed limits (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110 km/h)
- Prohibitory (no entry, no U-turn, no overtaking, no parking, no stopping, no horn)
- Warning (bend, curve, junction, roundabout, speed bump, pedestrian, school, animal, slippery)
- Mandatory (turn left, turn right, go straight, keep left, roundabout)
- Special (stop, give way)

Create: sign_database.json with all entries.
For TTS descriptions, use natural spoken language like:
"Speed limit sixty kilometers per hour. Please slow down."
NOT: "Speed limit 60 km/h"
```

---

### Member 4 — ML & Evaluation Lead

#### Prompt: YOLO Training on Google Colab
```
I need to train a YOLOv8-nano model on Google Colab (free GPU) for Malaysian road sign 
detection and then export it for mobile deployment.

Context:
- Dataset: ~300 images of Malaysian road signs, annotated in YOLO format
- Number of classes: ~50 (different sign types)
- Dataset is uploaded to Google Drive
- Need to export the trained model in ONNX format for later conversion to ncnn

Please create a complete Google Colab notebook (.ipynb style as code cells) that:
1. Installs ultralytics
2. Mounts Google Drive
3. Verifies dataset structure and data.yaml
4. Trains YOLOv8n with these settings:
   - epochs=100, imgsz=640, batch=16
   - patience=20 (early stopping)
   - augmentation enabled (mosaic, mixup, hsv adjustments)
5. Shows training metrics (mAP, loss curves)
6. Runs validation and shows confusion matrix
7. Exports best model to ONNX format (simplified)
8. Saves everything to Google Drive
9. Tests inference on a few sample images and displays results

Include instructions for:
- How to convert ONNX to ncnn format (using onnx2ncnn tool)
- What mAP score to aim for (>0.8 is good for this task)
- How to interpret the confusion matrix
- What to do if accuracy is low (more data, more augmentation, adjust hyperparams)

I'm new to deep learning. Please explain what each parameter means.
```

#### Prompt: Performance Testing & Evaluation
```
I need to create a comprehensive testing and evaluation pipeline for a YOLOv8-nano 
traffic sign detection model.

Requirements:
1. Test on all 84 provided test images and record:
   - Per-image: detected signs, confidence scores, inference time
   - Overall: accuracy, precision, recall, F1-score, mAP@0.5
2. Generate a confusion matrix visualization
3. Identify failure cases: which signs were missed or misclassified and analyze why
4. Test inference speed: measure average ms per image
5. Test beyond the 84 images on additional collected images
6. Compare two feature sets (HOG + color histogram vs deep learning features)
7. Create a results summary table suitable for the final report

Create Python scripts that:
1. test_model.py - runs inference on all test images, saves results to CSV
2. evaluate.py - calculates metrics and generates confusion matrix plot
3. compare_features.py - compares HOG+SVM vs YOLO recognition rates
4. Generate report-ready plots (matplotlib, saved as PNG)

Output should be formatted for direct inclusion in the FYP2 report.
```

---

## 🔧 PC Demo Setup

For presentations and supervisor demos, you need to show the app on a big screen:

### Option 1: scrcpy (Recommended — Easiest)
```bash
# 1. Install scrcpy (free, open source)
# Download from: https://github.com/Genymobile/scrcpy/releases
# Extract the zip, no installation needed

# 2. Connect Android phone via USB
# Enable USB Debugging in phone Settings → Developer Options

# 3. Run scrcpy
scrcpy.exe

# Your phone screen is now mirrored to PC with low latency
# The camera and app work on the real phone, screen shows on PC
```

### Option 2: Android Studio Emulator
```
1. Open Android Studio → Device Manager → Create Virtual Device
2. Choose a phone (e.g., Pixel 7)
3. For camera: in emulator settings, set Camera → Back → Webcam0
4. This uses your PC's webcam as the emulator's camera
5. Limitations: webcam quality may be lower, emulator is slower
```

### For Video Presentations
- Use **OBS Studio** (free) to record the scrcpy window
- Or use Android's built-in screen recording + voiceover

---

## ⚠️ Beginner Learning Path (First 2 Weeks)

Since the team is new to Android and deep learning, here's what each member should learn first:

| Member | Watch/Read First | Time |
|--------|-----------------|------|
| **Member 1** | [Android Basics in Kotlin](https://developer.android.com/courses/android-basics-compose/course) — Units 1-2 only | 4-6 hours |
| **Member 1** | [CameraX Getting Started](https://developer.android.com/codelabs/camerax-getting-started) — Google Codelab | 2 hours |
| **Member 2** | [Android NDK Getting Started](https://developer.android.com/ndk/guides) — just the basics | 2 hours |
| **Member 2** | [ncnn Wiki](https://github.com/Tencent/ncnn/wiki) — "how-to-use-ncnn-with-alexnet" example | 3 hours |
| **Member 3** | [Roboflow Annotation Tutorial](https://docs.roboflow.com/annotate) — free tier | 1 hour |
| **Member 3** | [Albumentations Tutorial](https://albumentations.ai/docs/getting_started/image_augmentation/) | 1 hour |
| **Member 4** | [Ultralytics YOLOv8 Quickstart](https://docs.ultralytics.com/quickstart/) | 2 hours |
| **Member 4** | [Google Colab Intro](https://colab.research.google.com/notebooks/intro.ipynb) | 1 hour |

---

*Last updated: 2026-06-24*
*Plan version: 2.0 — Finalized for Option B (Blind-Friendly Audio Assistant)*
