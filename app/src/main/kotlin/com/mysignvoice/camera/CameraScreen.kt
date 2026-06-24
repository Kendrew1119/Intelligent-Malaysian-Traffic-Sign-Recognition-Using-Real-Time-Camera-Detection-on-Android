// ============================================
// [Member 1] CameraScreen.kt
// ============================================
// Module: Camera Screen with Live Preview
// Owner: Member 1 (Android & UI Lead)
//
// Purpose:
//   - Full-screen camera preview using CameraX
//   - Overlay layer to draw bounding boxes on detected signs
//   - Bottom bar with Start/Stop toggle button (extra-large, 72dp+)
//   - Pass camera frames to native C++ library via NativeLib
//
// Key Dependencies:
//   - androidx.camera:camera-camera2
//   - androidx.camera:camera-lifecycle
//   - androidx.camera:camera-view
//   - Jetpack Compose
//
// References:
//   - Google CameraX Codelab: https://developer.android.com/codelabs/camerax-getting-started
//   - See plan.md → Member 1 Prompt: CameraX Setup
//
// TODO: Implement CameraX Preview + ImageAnalysis use cases
// ============================================

package com.mysignvoice.camera
