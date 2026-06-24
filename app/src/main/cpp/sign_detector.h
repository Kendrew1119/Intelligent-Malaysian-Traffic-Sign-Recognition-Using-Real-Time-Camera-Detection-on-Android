// ============================================
// [Member 2] sign_detector.h
// ============================================
// Module: Sign Detector Header (C++)
// Owner: Member 2 (Systems Architect)
//
// Purpose:
//   - Define Detection struct: { x, y, w, h, classId, confidence }
//   - Define SignDetector class with:
//       bool loadModel(const char* paramPath, const char* binPath)
//       std::vector<Detection> detect(cv::Mat& frame)
//       void setConfidenceThreshold(float threshold)
//       void setNmsThreshold(float threshold)
//   - ncnn::Net member variable
//   - OpenCV Mat for image processing
//
// Architecture:
//   Input (cv::Mat BGR) → resize to 640x640 → normalize → ncnn inference 
//   → parse raw output → NMS filtering → vector<Detection>
//
// References:
//   - ncnn API: https://github.com/Tencent/ncnn/wiki
//   - See plan.md → Member 2 Prompt: ncnn YOLOv8 Inference Engine
//   - See plan.md → ncnn Integration Step-by-Step Guide
//
// TODO: Implement Detection struct and SignDetector class declaration
// ============================================

#pragma once
