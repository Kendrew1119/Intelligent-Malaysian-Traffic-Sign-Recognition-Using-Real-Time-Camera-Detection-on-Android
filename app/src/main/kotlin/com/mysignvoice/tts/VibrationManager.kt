// ============================================
// [Member 1] VibrationManager.kt
// ============================================
// Module: Vibration Alert Manager
// Owner: Member 1 (Android & UI Lead)
//
// Purpose:
//   - Trigger different vibration patterns based on sign severity
//   - Patterns:
//       INFO signs (blue/mandatory):    single short buzz (100ms)
//       CAUTION signs (yellow/warning): two short buzzes (100ms-pause-100ms)
//       DANGER signs (red/prohibitory): one long buzz (500ms)
//       STOP sign:                      three rapid buzzes
//   - Respect system vibration settings
//   - Can be toggled on/off from settings
//
// Usage:
//   vibrationManager.vibrateForSign(severity="danger")
//
// References:
//   - Android Vibrator: https://developer.android.com/reference/android/os/Vibrator
//
// TODO: Implement vibration patterns
// ============================================

package com.mysignvoice.tts
