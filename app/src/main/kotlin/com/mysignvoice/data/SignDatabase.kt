// ============================================
// [Member 3] SignDatabase.kt
// ============================================
// Module: Sign Meaning Database
// Owner: Member 3 (Data & Database Lead)
//
// Purpose:
//   - Load sign_database.json from assets folder
//   - Provide lookup: getSign(classId: Int) → SignInfo
//   - SignInfo contains:
//       signId, nameEn, nameBm, descriptionEn, descriptionBm,
//       category (prohibitory/warning/mandatory/information),
//       severity (info/caution/danger),
//       shape (circle/triangle/rectangle/octagon/diamond),
//       color (red/blue/yellow/green)
//   - Used by TTSManager to get spoken description
//   - Used by VibrationManager to determine vibration pattern
//
// Data Source:
//   - app/src/main/assets/sign_database.json (created by Member 3)
//
// References:
//   - See plan.md → Member 3 Prompt: Sign Meaning Database
//
// TODO: Implement JSON parser and lookup functions
// ============================================

package com.mysignvoice.data
