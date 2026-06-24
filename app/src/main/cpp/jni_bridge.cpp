// ============================================
// [Member 2] jni_bridge.cpp
// ============================================
// Module: JNI Bridge (C++ Side)
// Owner: Member 2 (Systems Architect)
//
// Purpose:
//   - JNI functions that connect Kotlin NativeLib.kt ↔ C++ SignDetector
//   - Functions:
//       Java_com_mysignvoice_native_NativeLib_initModel()
//         → Create SignDetector, load .param and .bin from assets
//       Java_com_mysignvoice_native_NativeLib_detect()
//         → Receive pixel array from Kotlin, create cv::Mat, call detect()
//         → Convert vector<Detection> to Java/Kotlin array, return
//       Java_com_mysignvoice_native_NativeLib_setConfidenceThreshold()
//         → Update threshold on SignDetector instance
//       Java_com_mysignvoice_native_NativeLib_release()
//         → Delete SignDetector, free memory
//
// JNI Data Passing:
//   - Kotlin sends: jintArray (pixel data), jint width, jint height
//   - C++ receives: jint* → construct cv::Mat(height, width, CV_8UC4, pixels)
//   - C++ returns: jobjectArray of Detection objects
//
// Common Pitfalls:
//   - Always release JNI arrays with ReleaseIntArrayElements()
//   - Always release JNI strings with ReleaseStringUTFChars()
//   - Use GetPrimitiveArrayCritical for better performance
//
// References:
//   - JNI Guide: https://docs.oracle.com/javase/8/docs/technotes/guides/jni/
//   - See plan.md → Member 2 Prompt: JNI Bridge
//
// TODO: Implement all JNI functions
// ============================================

#include <jni.h>
#include "sign_detector.h"
