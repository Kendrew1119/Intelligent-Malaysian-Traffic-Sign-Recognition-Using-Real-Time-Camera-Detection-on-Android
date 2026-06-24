// ============================================
// [Member 1] TTSManager.kt
// ============================================
// Module: Text-to-Speech Manager
// Owner: Member 1 (Android & UI Lead)
//
// Purpose:
//   - Initialize Android TextToSpeech engine
//   - speakSign(signId, signName, description) function
//   - Language switching: English (en-MY) ↔ Bahasa Melayu (ms-MY)
//   - Detection suppression: don't repeat same sign within 5 seconds
//     (track lastDetectedClassId + lastDetectedTimestamp)
//   - Queue management: interrupt current speech for new detection
//   - Shutdown TTS engine on app close
//
// Usage:
//   ttsManager.speakSign(signId=12, name="Speed 60", desc="Speed limit 60 km/h ahead")
//   ttsManager.setLanguage("ms-MY")
//   ttsManager.stop()
//
// References:
//   - Android TTS: https://developer.android.com/reference/android/speech/tts/TextToSpeech
//   - See plan.md → Member 1 Prompt: Text-to-Speech Implementation
//
// TODO: Implement TTS lifecycle and speech functions
// ============================================

package com.mysignvoice.tts
