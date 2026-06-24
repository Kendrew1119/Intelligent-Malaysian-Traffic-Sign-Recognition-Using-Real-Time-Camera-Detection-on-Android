// ============================================
// [Member 2] Detection.kt
// ============================================
// Module: Detection Data Class
// Owner: Member 2 (Systems Architect)
//
// Purpose:
//   - Data class matching the C++ Detection struct
//   - Represents one detected road sign in a frame
//
// Fields:
//   - x: Float          → top-left x of bounding box (normalized 0-1)
//   - y: Float          → top-left y of bounding box (normalized 0-1)
//   - width: Float      → width of bounding box (normalized 0-1)
//   - height: Float     → height of bounding box (normalized 0-1)
//   - classId: Int       → sign class ID (maps to sign_database.json)
//   - confidence: Float  → detection confidence score (0.0 - 1.0)
//   - label: String      → sign name (filled by Kotlin after lookup)
//
// TODO: Implement data class
// ============================================

package com.mysignvoice.native
