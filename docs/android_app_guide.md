# 📱 Option 1: Android App Framework Integration Guide

This guide outlines the steps to build the final Android app (due Week 13) that will use your trained YOLOv8 model to detect traffic signs in real-time.

---

## 🏗️ Architecture Overview

The Android app combines three major technologies:
1. **Kotlin & Jetpack Compose** (Frontend/UI)
2. **CameraX** (Camera feed)
3. **C++ (JNI) & Tencent ncnn** (Backend/AI Inference)

---

## 📝 Task Distribution (For Team Members)

To complete the app efficiently, the workload should be distributed as follows:

| Role | Member Focus | Responsibilities |
|------|--------------|------------------|
| **UI/UX & TTS** | Member 1 & 2 | Build the Jetpack Compose screens (Home, Camera). Implement Text-to-Speech (TTS) so the app speaks the detected signs (e.g. "School zone ahead"). |
| **CameraX setup** | Member 3 | Setup the Android CameraX API to capture frames from the phone's back camera and pass them to the C++ layer. |
| **C++ JNI (AI)** | Member 4 | Write the `yolo.cpp` bridge using JNI to load `yolov8n_signs.param` and `yolov8n_signs.bin`, and run real-time inference on the camera frames. |

---

## 🛠️ Step-by-Step Implementation Guide

### Step 1: Create the Android Studio Project
1. Open Android Studio -> **New Project**.
2. Select **Empty Compose Activity** (to use modern Jetpack Compose UI).
3. Name: `IntelligentTrafficSignRecognition`.
4. Language: **Kotlin**.
5. Minimum SDK: **API 24 (Android 7.0)**.

### Step 2: Add Native C++ Support
1. Right-click the `app` folder -> **Add C++ to Module**.
2. This will generate a `CMakeLists.txt` and a `cpp` folder. 
3. This is where Member 4 will write the JNI bridge code.

### Step 3: Import the ncnn Library and Your Model
1. Download the pre-compiled `ncnn-android-vulkan` release from the official Tencent ncnn GitHub.
2. Put the ncnn library files into your `app/src/main/cpp/ncnn/` folder.
3. Update `CMakeLists.txt` to link the ncnn library to your project.
4. **Important:** Move your newly downloaded `yolov8n_signs.param` and `yolov8n_signs.bin` into the `app/src/main/assets/` directory. Android needs them in the assets folder to bundle them with the app.

### Step 4: Setup CameraX (Member 3)
1. Add CameraX dependencies in your `build.gradle.kts`.
2. Request `CAMERA` permissions in `AndroidManifest.xml`.
3. Create an `ImageAnalysis.Analyzer` class that receives frames from the camera.
4. Convert the YUV camera frame to an RGB Bitmap, and pass the Bitmap to the C++ JNI function.

### Step 5: Write the JNI Bridge (Member 4)
1. In `app/src/main/cpp/native-lib.cpp`, write a function that takes the Bitmap and runs ncnn inference.
```cpp
// Example pseudocode for the C++ side
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_trafficsign_Yolo_detect(JNIEnv* env, jobject thiz, jobject bitmap) {
    // 1. Convert bitmap to ncnn::Mat
    // 2. ncnn::Extractor ex = yolo_net.create_extractor();
    // 3. ex.input("in0", in);
    // 4. ex.extract("out0", out);
    // 5. Parse bounding boxes and classes
    // 6. Return array of detected objects back to Kotlin
}
```

### Step 6: Display & Speak Results (Members 1 & 2)
1. In Kotlin, receive the array of detected objects from C++.
2. Draw a rectangle over the camera preview at the bounding box coordinates.
3. Use Android's `TextToSpeech` engine to announce new detections:
   ```kotlin
   textToSpeech.speak("Speed limit sign ahead", TextToSpeech.QUEUE_FLUSH, null, "")
   ```

---

## 🚀 Next Actions
If you decide to start this option now, we will begin by generating the Android Studio project scaffold and the `CMakeLists.txt` configuration for ncnn.
