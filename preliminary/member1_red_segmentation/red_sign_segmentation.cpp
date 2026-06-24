// ============================================
// [Member 1] red_sign_segmentation.cpp
// ============================================
// Module: Red Sign Segmentation Using Color Information
// Owner: Member 1
// Due: Week 6-7 (Preliminary Work, Chapter 4)
//
// Purpose:
//   - Segment RED traffic signs from the 84 test images
//   - Uses OpenCV C++ with HSV color thresholding
//
// Algorithm:
//   1. Read input image: cv::imread(imagePath)
//   2. Convert BGR to HSV: cv::cvtColor(src, hsv, COLOR_BGR2HSV)
//   3. Threshold for RED hue range:
//      - Red wraps around H=0 and H=180 in OpenCV HSV
//      - Lower red: H=[0, 10],   S=[70, 255], V=[50, 255]
//      - Upper red: H=[170, 180], S=[70, 255], V=[50, 255]
//      - cv::inRange(hsv, lower1, upper1, mask1)
//      - cv::inRange(hsv, lower2, upper2, mask2)
//      - mask = mask1 | mask2
//   4. Morphological operations to clean up:
//      - cv::erode → remove noise
//      - cv::dilate → fill gaps
//      - cv::morphologyEx with MORPH_OPEN and MORPH_CLOSE
//   5. Find contours: cv::findContours
//   6. Draw bounding rectangles around detected red regions
//   7. Save output image showing segmented regions
//
// Test Images:
//   Read from: ../Color Inputs/Red Signs/
//
// Build (Windows, Visual Studio):
//   cl /EHsc red_sign_segmentation.cpp /I "C:\opencv\include" /link /LIBPATH:"C:\opencv\lib" opencv_world4xx.lib
//
// TODO: Implement red sign segmentation
// ============================================

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // TODO: Implement
    std::cout << "Red Sign Segmentation - Member 1" << std::endl;
    return 0;
}
