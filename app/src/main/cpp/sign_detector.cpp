// ============================================
// [Member 2] sign_detector.cpp
// ============================================
// Module: Sign Detector Implementation (C++)
// Owner: Member 2 (Systems Architect)
//
// Purpose:
//   - Implement SignDetector::loadModel()
//       Load ncnn .param and .bin files
//       Configure: num_threads=4, use_vulkan=false
//   - Implement SignDetector::detect()
//       1. Resize input Mat to 640x640 using cv::resize
//       2. Convert BGR to RGB
//       3. Create ncnn::Mat from pixel data
//       4. Normalize: substract_mean_normalize(mean={0,0,0}, norm={1/255,1/255,1/255})
//       5. Run ncnn extractor: input("in0") → extract("out0")
//       6. Parse output tensor into Detection structs
//       7. Apply Non-Maximum Suppression (NMS) with IoU threshold 0.45
//       8. Filter by confidence threshold (default 0.5)
//       9. Return vector<Detection>
//
// Key ncnn API Calls:
//   ncnn::Net net;
//   net.load_param("model.param");
//   net.load_model("model.bin");
//   ncnn::Extractor ex = net.create_extractor();
//   ex.input("in0", input_mat);
//   ex.extract("out0", output_mat);
//
// References:
//   - ncnn YOLOv8 example: https://github.com/Tencent/ncnn/tree/master/examples
//   - See plan.md → ncnn Integration Guide → Step 4
//
// TODO: Implement full inference pipeline
// ============================================

#include "sign_detector.h"
