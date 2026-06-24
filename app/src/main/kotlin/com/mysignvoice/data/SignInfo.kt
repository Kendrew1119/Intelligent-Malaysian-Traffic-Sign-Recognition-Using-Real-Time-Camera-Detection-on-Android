// ============================================
// [Member 3] SignInfo.kt
// ============================================
// Module: Sign Information Data Class
// Owner: Member 3 (Data & Database Lead)
//
// Purpose:
//   - Data class representing a road sign's full information
//   - Parsed from sign_database.json
//
// Fields:
//   signId: Int                → YOLO class ID
//   nameEn: String             → English name (e.g., "Speed Limit 60")
//   nameBm: String             → Bahasa Melayu name (e.g., "Had Laju 60")
//   descriptionEn: String      → TTS-friendly English description
//   descriptionBm: String      → TTS-friendly BM description
//   category: String           → "prohibitory" | "warning" | "mandatory" | "information"
//   severity: String           → "info" | "caution" | "danger"
//   shape: String              → "circle" | "triangle" | "rectangle" | "octagon" | "diamond"
//   color: String              → "red" | "blue" | "yellow" | "green"
//
// TODO: Implement data class
// ============================================

package com.mysignvoice.data
