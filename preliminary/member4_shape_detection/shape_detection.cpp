// ============================================
// [Member 4] shape_detection.cpp
// ============================================
// Module: Shape Detection of Traffic Signs
// Owner: Member 4
// Due: Week 6-7 (Preliminary Work, Chapter 4)
//
// Purpose:
//   - Detect and classify SHAPES of traffic signs
//   - Shapes: circle, triangle, rectangle, octagon
//   - Uses OpenCV C++ edge detection + contour analysis
//
// Algorithm:
//   1. Read input image: cv::imread(imagePath)
//   2. Convert to grayscale: cv::cvtColor(src, gray, COLOR_BGR2GRAY)
//   3. Apply Gaussian blur: cv::GaussianBlur (reduce noise)
//   4. Edge detection: cv::Canny(gray, edges, 50, 150)
//   5. Find contours: cv::findContours(edges, contours, ...)
//   6. For each contour:
//      a. Approximate polygon: cv::approxPolyDP(contour, approx, epsilon, true)
//      b. Classify shape by vertex count:
//         - 3 vertices → Triangle (warning sign)
//         - 4 vertices → Rectangle (info sign) or Square
//         - 8 vertices → Octagon (stop sign)
//         - >8 vertices → Circle (speed limit / prohibitory)
//      c. Verify with:
//         - cv::isContourConvex (should be convex for signs)
//         - cv::contourArea (filter too small/large)
//         - Aspect ratio check
//   7. Draw detected shapes with labels
//   8. Save output image
//
// Test Images:
//   Read from: ../Color Inputs/ (all subdirectories)
//
// Build (Windows, Visual Studio):
//   cl /EHsc shape_detection.cpp /I "C:\opencv\include" /link /LIBPATH:"C:\opencv\lib" opencv_world4xx.lib
//
// TODO: Implement shape detection and classification
// ============================================

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // TODO: Implement
    std::cout << "Shape Detection - Member 4" << std::endl;
    return 0;
}
