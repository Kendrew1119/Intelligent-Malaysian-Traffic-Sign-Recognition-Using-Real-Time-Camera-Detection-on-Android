// ============================================
// [Member 3] DetectionHistoryRepository.kt
// ============================================
// Module: Detection History Repository
// Owner: Member 3 (Data & Database Lead)
//
// Purpose:
//   - Store last 50 detection records locally
//   - Each record: timestamp, signId, signName, confidence
//   - Uses Room database or SharedPreferences
//   - Functions:
//       addDetection(signId, signName, confidence, timestamp)
//       getHistory(): List<DetectionRecord>
//       clearHistory()
//
// Used By:
//   - HistoryScreen.kt (Member 1) to display past detections
//   - CameraFrameAnalyzer.kt (Member 1) to save new detections
//
// TODO: Implement Room database or SharedPreferences storage
// ============================================

package com.mysignvoice.data
