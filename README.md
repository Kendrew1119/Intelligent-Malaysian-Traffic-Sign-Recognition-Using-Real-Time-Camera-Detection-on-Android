# MYSignVoice 🦯

**Intelligent Malaysian Traffic Sign Recognition for Visually Impaired Pedestrians Using Real-Time Camera Detection on Android**

## Overview

An Android app that uses the phone camera to detect Malaysian road signs in real-time and speaks their meaning aloud using Text-to-Speech (TTS) for visually impaired pedestrians.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core Logic | C++17 |
| Android App | Kotlin + Jetpack Compose |
| Camera | CameraX |
| Object Detection | YOLOv8-nano |
| Inference Engine | ncnn (C++) |
| Image Processing | OpenCV (C++) |
| TTS | Android TextToSpeech API |
| Training | Ultralytics + Google Colab |

## Project Structure

```
miniproject/
├── app/                              # Android Application
│   ├── src/main/
│   │   ├── kotlin/com/mysignvoice/
│   │   │   ├── camera/               # [Member 1] CameraX integration
│   │   │   ├── ui/                   # [Member 1] Jetpack Compose screens
│   │   │   ├── tts/                  # [Member 1] Text-to-Speech & vibration
│   │   │   ├── native/               # [Member 2] JNI bridge (Kotlin side)
│   │   │   ├── data/                 # [Member 3] Sign database & history
│   │   │   └── ml/                   # [Member 4] Model manager & evaluation
│   │   ├── cpp/                      # [Member 2] C++ native library
│   │   │   ├── sign_detector.h/.cpp  # ncnn YOLO inference
│   │   │   ├── jni_bridge.cpp        # JNI functions
│   │   │   └── CMakeLists.txt        # C++ build config
│   │   ├── assets/                   # Model files & sign database JSON
│   │   └── res/                      # Android resources
├── dataset/                          # [Member 3] Training data
├── training/                         # [Member 4] Training & evaluation scripts
├── preliminary/                      # [All] Week 4-7 preliminary work (C++)
│   ├── member1_red_segmentation/
│   ├── member2_blue_segmentation/
│   ├── member3_yellow_segmentation/
│   └── member4_shape_detection/
└── docs/                             # Reports & presentations
```

## Setup

1. Copy `.env.example` to `.env` and fill in your local paths
2. Read `plan.md` for the full project plan
3. Each member works on their assigned folder/module

## Team

| Member | Role | Modules |
|--------|------|---------|
| Member 1 | Android & UI Lead | camera/, ui/, tts/ |
| Member 2 | Systems Architect | cpp/, native/ |
| Member 3 | Data & Database Lead | dataset/, data/, assets/ |
| Member 4 | ML & Evaluation Lead | training/, ml/ |

## License

Academic project for UCCC2513 Mini Project.
