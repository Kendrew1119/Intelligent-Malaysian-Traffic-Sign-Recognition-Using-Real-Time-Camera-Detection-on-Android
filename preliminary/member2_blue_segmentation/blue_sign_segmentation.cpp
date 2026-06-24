// ============================================
// [Member 2] blue_sign_segmentation.cpp
// ============================================
// Module: Blue Sign Segmentation Using Color Information
// Owner: Member 2
// Due: Week 6-7 (Preliminary Work, Chapter 4)
//
// Purpose:
//   - Segment BLUE traffic signs from the 84 test images
//   - Uses OpenCV C++ with HSV color thresholding
//
// Algorithm:
//   1. Read input image: cv::imread(imagePath)
//   2. Convert BGR to HSV: cv::cvtColor(src, hsv, COLOR_BGR2HSV)
//   3. Threshold for BLUE hue range:
//      - Blue: H=[100, 130], S=[50, 255], V=[50, 255]
//      - cv::inRange(hsv, lower, upper, mask)
//   4. Morphological operations:
//      - cv::erode → remove noise
//      - cv::dilate → fill gaps
//      - cv::morphologyEx (MORPH_OPEN, MORPH_CLOSE)
//   5. Find contours: cv::findContours
//   6. Filter contours by area (remove too small/large)
//   7. Draw bounding rectangles around blue regions
//   8. Save output image
//
// Test Images:
//   Read from: ../Color Inputs/Blue Signs/
//
// Build (Windows, Visual Studio):
//   cl /EHsc blue_sign_segmentation.cpp /I "C:\opencv\include" /link /LIBPATH:"C:\opencv\lib" opencv_world4xx.lib
//
// TODO: Implement blue sign segmentation
// ============================================

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // TODO: Implement
    std::cout << "Blue Sign Segmentation - Member 2" << std::endl;
    return 0;
}
