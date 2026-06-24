// ============================================
// [Member 2] NativeLib.kt
// ============================================
// Module: JNI Bridge (Kotlin Side)
// Owner: Member 2 (Systems Architect)
//
// Purpose:
//   - Kotlin class that declares external (native) C++ functions
//   - Loads the "signdetector" shared library (.so)
//   - Provides Kotlin-friendly API for:
//       initModel(paramPath, binPath) → Boolean
//       detect(bitmapPixels, width, height) → Array<Detection>
//       setConfidenceThreshold(threshold: Float)
//       release() → cleanup native resources
//
// JNI Flow:
//   Kotlin calls NativeLib.detect(pixels) 
//     → JNI marshals data to C++ 
//     → C++ runs OpenCV + ncnn inference 
//     → C++ returns results via JNI 
//     → Kotlin receives Array<Detection>
//
// References:
//   - Android JNI Guide: https://developer.android.com/ndk/guides
//   - See plan.md → Member 2 Prompt: JNI Bridge
//
// TODO: Implement external function declarations + companion object with System.loadLibrary
// ============================================

package com.mysignvoice.native
