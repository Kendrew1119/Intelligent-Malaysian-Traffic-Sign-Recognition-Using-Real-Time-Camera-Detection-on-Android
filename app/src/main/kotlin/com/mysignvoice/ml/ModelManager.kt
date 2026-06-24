// ============================================
// [Member 4] ModelManager.kt
// ============================================
// Module: Model Manager
// Owner: Member 4 (ML & Evaluation Lead)
//
// Purpose:
//   - Manage YOLO model lifecycle on Android
//   - Copy model files (.param, .bin) from assets to internal storage on first launch
//   - Provide model file paths to NativeLib for initialization
//   - Handle model versioning (if model is updated)
//   - Track inference performance metrics (avg ms per frame)
//
// Model Files (stored in assets/models/):
//   - yolov8n_signs.param  (~50 KB, model architecture)
//   - yolov8n_signs.bin    (~6 MB, model weights)
//
// Usage:
//   val modelManager = ModelManager(context)
//   val (paramPath, binPath) = modelManager.getModelPaths()
//   nativeLib.initModel(paramPath, binPath)
//
// TODO: Implement model file management
// ============================================

package com.mysignvoice.ml
