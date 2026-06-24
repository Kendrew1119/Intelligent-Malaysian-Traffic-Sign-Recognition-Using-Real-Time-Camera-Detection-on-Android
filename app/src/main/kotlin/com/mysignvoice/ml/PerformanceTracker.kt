// ============================================
// [Member 4] PerformanceTracker.kt
// ============================================
// Module: Performance Tracker
// Owner: Member 4 (ML & Evaluation Lead)
//
// Purpose:
//   - Track inference time per frame (milliseconds)
//   - Calculate rolling average FPS
//   - Log detection accuracy metrics during testing
//   - Provide performance data for the final report:
//       - Average inference time
//       - Min/max inference time
//       - Frames per second
//       - Detection count per sign class
//   - Optional: display FPS counter on debug overlay
//
// Usage:
//   performanceTracker.startFrame()
//   // ... inference happens ...
//   performanceTracker.endFrame(numDetections)
//   val avgMs = performanceTracker.getAverageInferenceTime()
//
// TODO: Implement performance tracking
// ============================================

package com.mysignvoice.ml
