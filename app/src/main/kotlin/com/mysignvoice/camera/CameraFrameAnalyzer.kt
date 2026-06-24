// ============================================
// [Member 1] CameraFrameAnalyzer.kt
// ============================================
// Module: Camera Frame Analyzer
// Owner: Member 1 (Android & UI Lead)
//
// Purpose:
//   - Implements ImageAnalysis.Analyzer interface
//   - Receives camera frames as ImageProxy at 15-30 FPS
//   - Converts ImageProxy → Bitmap → passes to NativeLib.detect()
//   - Handles frame rotation correction
//   - Throttles frames if inference is slower than capture rate
//
// Key Flow:
//   ImageProxy → YUV to RGB conversion → Bitmap → NativeLib.detect() → List<Detection>
//
// References:
//   - See plan.md → Member 1 Prompt: CameraX Setup
//
// TODO: Implement ImageAnalysis.Analyzer
// ============================================

package com.mysignvoice.camera
