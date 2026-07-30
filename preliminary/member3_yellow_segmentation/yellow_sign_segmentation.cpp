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
//   - Implements the 6-panel grid layout ("follow the partition") 
//     matching the format of Member 4's shape detection
//
// Approach:
//   1. Convert image to HSV color space
//   2. Apply yellow-specific mask (H=[12, 38], S=[80, 255], V=[50, 255])
//   3. Morphological OPEN + CLOSE to clean mask
//   4. Find external contours on the clean mask
//   5. Select the largest contour as the sign boundary
//   6. Classify shape using minEnclosingCircle circularity + vertex count
//   7. Display 6-panel grid and save output
//
// Test Images:
//   Read from: ../../Color Inputs/Yellow Signs/
//
// Build (Windows, Visual Studio):
//   cl /EHsc yellow_sign_segmentation.cpp /I "C:\opencv\include" /link /LIBPATH:"C:\opencv\lib" opencv_world4xx.lib
// ============================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

// ============================================
// Function: getYellowMask
// ============================================
// Creates a binary mask that isolates yellow pixels
// using HSV thresholding. Applies morphological
// OPEN and CLOSE to filter noise and fill gaps.
// ============================================
cv::Mat getYellowMask(const cv::Mat& src) {
    cv::Mat hsv, mask;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // Yellow HSV threshold:
    // H: [12, 38] -> covers amber/yellow range
    // S: [80, 255] -> filters out desaturated backgrounds
    // V: [50, 255] -> filters out shadows and low-light regions
    cv::inRange(hsv, cv::Scalar(12, 80, 50), cv::Scalar(38, 255, 255), mask);

    // Morphological OPEN (3x3 rect kernel) to remove small noise dots
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    
    // Morphological CLOSE (3x3 rect kernel) to fill hollow parts of the sign
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    return mask;
}

// ============================================
// Function: classifyShape
// ============================================
// Classifies shape using circularity and polygon
// approximation. Leverages loose (4% perimeter) and
// strict (1% perimeter) epsilon parameters.
// ============================================
std::string classifyShape(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double peri = cv::arcLength(contour, true);

    std::vector<cv::Point> approxLoose, approxStrict;
    // Loose approximation for triangle (3) and rectangle (4)
    cv::approxPolyDP(contour, approxLoose, 0.04 * peri, true);
    // Strict approximation for octagon (7-9) and circle (>9)
    cv::approxPolyDP(contour, approxStrict, 0.01 * peri, true);

    int verticesLoose = (int)approxLoose.size();
    int verticesStrict = (int)approxStrict.size();

    // Circularity using minimum enclosing circle area ratio
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contour, center, radius);
    double enclosingArea = CV_PI * radius * radius;
    double circularity = (enclosingArea > 0) ? (area / enclosingArea) : 0;

    if (verticesLoose == 3) {
        return "Triangle";
    }
    else if (verticesLoose == 4) {
        return "Rectangle";
    }
    else if (circularity > 0.75) {
        if (verticesStrict >= 7 && verticesStrict <= 9) {
            return "Octagon";
        }
        else {
            return "Circle";
        }
    }
    else {
        return "Polygon";
    }
}

// ============================================
// Function: processImage
// ============================================
// Performs segmentation, shapes classification,
// and produces the standard 6-panel grid.
// ============================================
std::string processImage(const cv::Mat& src, const std::string& filename,
    const std::string& outPath, bool showSteps) {

    // Resize image to fixed 300x300 for consistent layout
    cv::Mat img;
    cv::resize(src, img, cv::Size(300, 300));
    cv::Mat black = cv::Mat::zeros(img.size(), img.type());

    // 1. Get binary mask for yellow pixels
    cv::Mat yellowMask = getYellowMask(img);

    // 2. Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(yellowMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Initialize 6 panels
    cv::Mat p1 = img.clone();                    // Original Image
    cv::Mat p2 = black.clone();                  // All Contours
    cv::Mat p3 = black.clone();                  // Largest Contour boundary
    cv::Mat p4 = black.clone();                  // Clean Binary Mask (filled)
    cv::Mat p5 = black.clone();                  // Shape name classification (filled green)
    cv::Mat p6 = black.clone();                  // Segmented Yellow Sign (masked original)

    std::string shapeName = "No Shape";

    // Draw all detected contours in magenta on panel 2
    cv::drawContours(p2, contours, -1, cv::Scalar(255, 0, 255), 2);

    // 3. Find and isolate the largest yellow contour
    if (!contours.empty()) {
        int largestIdx = 0;
        double maxArea = 0;
        for (int i = 0; i < (int)contours.size(); i++) {
            double a = cv::contourArea(contours[i]);
            if (a > maxArea) {
                maxArea = a;
                largestIdx = i;
            }
        }

        // Apply a threshold of 500 pixels to eliminate noise
        if (maxArea > 500) {
            // Draw largest contour boundary on panel 3 (white outline)
            cv::drawContours(p3, contours, largestIdx, cv::Scalar(255, 255, 255), 2);

            // Create filled mask on panel 4 (white fill)
            cv::drawContours(p4, contours, largestIdx, cv::Scalar(255, 255, 255), cv::FILLED);

            // 4. Classify shape of yellow sign
            shapeName = classifyShape(contours[largestIdx]);

            // Draw filled green shape on panel 5
            cv::drawContours(p5, contours, largestIdx, cv::Scalar(0, 255, 0), cv::FILLED);

            // 5. Segment the sign (bitwise AND using the binary mask)
            cv::Mat maskGray;
            cv::cvtColor(p4, maskGray, cv::COLOR_BGR2GRAY);
            cv::bitwise_and(img, img, p6, maskGray);
        }
    }

    // 6. Add standard labels to the panels
    auto addLabel = [](cv::Mat& panel, const std::string& text) {
        int h = panel.rows;
        cv::putText(panel, text, cv::Point(5, h - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    };

    addLabel(p1, "Original");
    addLabel(p2, "All Contours");
    addLabel(p3, "Largest Contour");
    addLabel(p4, "Yellow Mask");
    addLabel(p5, "Shape Classification: " + shapeName);
    addLabel(p6, "Segmented Sign");

    // 7. Combine panels into a 3x2 grid (follow the partition)
    cv::Mat topRow, bottomRow, grid;
    cv::hconcat(std::vector<cv::Mat>{p1, p2, p3}, topRow);
    cv::hconcat(std::vector<cv::Mat>{p4, p5, p6}, bottomRow);
    cv::vconcat(topRow, bottomRow, grid);

    // Save grid result
    cv::imwrite(outPath, grid);

    // Optional interactive visualization
    if (showSteps) {
        std::string winName = "Yellow Sign Segmentation - " + filename;
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 900, 600);
        cv::imshow(winName, grid);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "  [" << filename << "] Detected: " << shapeName << std::endl;
    return shapeName;
}

// ============================================
// Main Function
// ============================================
int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << " Member 3: Yellow Sign Segmentation" << std::endl;
    std::cout << " MYSignVoice Preliminary Work" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check configuration arguments
    bool showSteps = false;
#ifdef _DEBUG
    showSteps = true;
    std::cout << "Mode: Debug mode detected (Visual Studio). Visualization enabled automatically." << std::endl;
#endif
    if (argc > 1 && std::string(argv[1]) == "--show") {
        showSteps = true;
        std::cout << "Mode: Step-by-step visualization enabled" << std::endl;
    }

    // Robust path lookup for the Color Inputs directory
    std::string baseDir = "";
    std::vector<std::string> pathCandidates = {
        "../../Color Inputs/Yellow Signs",
        "../Color Inputs/Yellow Signs",
        "Color Inputs/Yellow Signs"
    };

    for (const auto& candidate : pathCandidates) {
        if (fs::exists(candidate)) {
            baseDir = candidate;
            break;
        }
    }

    if (baseDir.empty()) {
        std::cerr << "ERROR: Could not locate directory 'Color Inputs/Yellow Signs'!" << std::endl;
        std::cerr << "Make sure the 'Color Inputs' folder is at the root level." << std::endl;
        return -1;
    }

    std::cout << "Reading images from: " << baseDir << std::endl;

    // Create output folder in the same directory as executable
    std::string outputDir = "output";
    fs::create_directories(outputDir);

    int totalImages = 0;
    int totalDetected = 0;
    int totalCircle = 0, totalTriangle = 0, totalRect = 0, totalOctagon = 0, totalPolygon = 0;

    for (const auto& entry : fs::directory_iterator(baseDir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        // Convert extension to lowercase for comparison
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") continue;

        std::string filepath = entry.path().string();
        std::string filename = entry.path().filename().string();

        cv::Mat img = cv::imread(filepath);
        if (img.empty()) {
            std::cerr << "  WARNING: Failed to read image " << filename << std::endl;
            continue;
        }

        totalImages++;

        // Define output file path
        std::string outputPath = outputDir + "/Grid_" + filename;

        // Process yellow sign segmentation
        std::string shape = processImage(img, filename, outputPath, showSteps);

        if (shape != "No Shape") {
            totalDetected++;
            if (shape == "Circle") totalCircle++;
            else if (shape == "Triangle") totalTriangle++;
            else if (shape == "Rectangle") totalRect++;
            else if (shape == "Octagon") totalOctagon++;
            else if (shape == "Polygon") totalPolygon++;
        }
    }

    // Output stats matching Member 4's reporting formatting
    double overallAcc = (totalImages > 0) ? (100.0 * totalDetected / totalImages) : 0;
    std::cout << "\n========================================" << std::endl;
    std::cout << " YELLOW SIGN SEGMENTATION SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Total images processed: " << totalImages << std::endl;
    std::cout << " Total shapes detected:  " << totalDetected << std::endl;
    std::cout << " Detection Rate:         " << std::fixed << std::setprecision(1) << overallAcc << "%" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << " Detected Shape Breakdown:" << std::endl;
    std::cout << "   Circle:    " << totalCircle << std::endl;
    std::cout << "   Triangle:  " << totalTriangle << std::endl;
    std::cout << "   Rectangle: " << totalRect << std::endl;
    std::cout << "   Octagon:   " << totalOctagon << std::endl;
    std::cout << "   Polygon:   " << totalPolygon << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Results saved to directory: " << outputDir << "/" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
