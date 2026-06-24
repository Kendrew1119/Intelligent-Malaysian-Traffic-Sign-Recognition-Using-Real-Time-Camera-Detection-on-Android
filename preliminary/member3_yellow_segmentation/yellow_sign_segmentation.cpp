// ============================================
// [Member 3] yellow_sign_segmentation.cpp
// ============================================
// Module: Yellow Sign Segmentation Using Color Information
// Owner: Member 3
// Due: Week 6-7 (Preliminary Work, Chapter 4)
//
// Purpose:
//   - Segment YELLOW traffic signs from the 84 test images
//   - Uses OpenCV C++ with HSV color thresholding
//
// Algorithm:
//   1. Read input image: cv::imread(imagePath)
//   2. Convert BGR to HSV: cv::cvtColor(src, hsv, COLOR_BGR2HSV)
//   3. Threshold for YELLOW hue range:
//      - Yellow: H=[15, 35], S=[80, 255], V=[80, 255]
//      - cv::inRange(hsv, lower, upper, mask)
//   4. Morphological operations:
//      - cv::erode → remove noise
//      - cv::dilate → fill gaps
//      - cv::morphologyEx (MORPH_OPEN, MORPH_CLOSE)
//   5. Find contours: cv::findContours
//   6. Filter contours by area
//   7. Draw bounding rectangles around yellow regions
//   8. Save output image
//
// Test Images:
//   Read from: ../Color Inputs/Yellow Signs/
//
// Build (Windows, Visual Studio):
//   cl /EHsc yellow_sign_segmentation.cpp /I "C:\opencv\include" /link /LIBPATH:"C:\opencv\lib" opencv_world4xx.lib
//
// TODO: Implement yellow sign segmentation
// ============================================

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // TODO: Implement
    std::cout << "Yellow Sign Segmentation - Member 3" << std::endl;
    return 0;
}
