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
//   - Uses OpenCV C++ HSV color segmentation + contour analysis
//
// Approach:
//   1. Convert image to HSV color space
//   2. Apply color-specific mask (Red/Blue/Yellow thresholds)
//   3. Morphological OPEN + CLOSE to clean mask
//   4. Find external contours on the clean mask
//   5. Select the largest contour as the sign boundary
//   6. Classify shape using minEnclosingCircle circularity + vertex count
//   7. Display 6-panel grid and save output
//
// Test Images:
//   Read from: ../../Color Inputs/ (Red Signs, Blue Signs, Yellow Signs)
//
// Build (Windows, Visual Studio 2022):
//   See docs/member4_shape_detection_guide.md
// ============================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

// ============================================
// Function: getColorMask
// ============================================
// Creates a binary mask that isolates sign-colored
// pixels using HSV thresholding. Different thresholds
// are used for each sign color category.
// ============================================
cv::Mat getColorMask(const cv::Mat& src, const std::string& colorType) {
    cv::Mat hsv, mask;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    if (colorType == "Red Signs") {
        // Red wraps around 0/180 in HSV, so we need two ranges
        cv::Mat mask1, mask2;
        cv::inRange(hsv, cv::Scalar(0, 65, 55), cv::Scalar(10, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(165, 60, 55), cv::Scalar(180, 255, 255), mask2);
        mask = mask1 | mask2;
    }
    else if (colorType == "Blue Signs") {
        cv::inRange(hsv, cv::Scalar(85, 102, 31), cv::Scalar(135, 255, 255), mask);
    }
    else if (colorType == "Yellow Signs") {
        cv::inRange(hsv, cv::Scalar(12, 80, 50), cv::Scalar(38, 255, 255), mask);
    }
    else {
        // Fallback: use saturation thresholding for any color
        cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(180, 255, 255), mask);
    }

    // Morphological OPEN to remove small noise spots
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    // Morphological CLOSE to fill small holes inside the sign region
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    return mask;
}

// ============================================
// Function: classifyShape
// ============================================
// Uses minEnclosingCircle circularity to distinguish
// circles from polygons, then vertex count for others.
// This approach (from the reference code) is more
// robust than using perimeter-based circularity alone.
// ============================================
std::string classifyShape(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double peri = cv::arcLength(contour, true);

    // Use two levels of polygon approximation
    // Loose: for triangle/rectangle detection (fewer vertices)
    std::vector<cv::Point> approxLoose, approxStrict;
    cv::approxPolyDP(contour, approxLoose, 0.04 * peri, true);
    // Strict: for octagon detection (more vertices preserved)
    cv::approxPolyDP(contour, approxStrict, 0.01 * peri, true);

    int verticesLoose = (int)approxLoose.size();
    int verticesStrict = (int)approxStrict.size();

    // Compute circularity using minEnclosingCircle
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contour, center, radius);
    double enclosingArea = CV_PI * radius * radius;
    double circularity = (enclosingArea > 0) ? (area / enclosingArea) : 0;

    // Classification logic (matching the reference sample code)
    if (verticesLoose == 3) {
        return "Triangle";
    }
    else if (verticesLoose == 4) {
        return "Rectangle";
    }
    else if (circularity > 0.75) {
        // High circularity = round shape
        // But check strict vertices to distinguish octagon from circle
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
// Main pipeline for one image.
// Creates a 6-panel grid output (matching the
// lecturer's reference format):
//   [Original | Contours | Largest Contour]
//   [Mask     | Shape    | Sign Segmented ]
// Returns the shape name detected (for statistics).
// ============================================
std::string processImage(const cv::Mat& src, const std::string& filename,
    const std::string& colorType, const std::string& outPath, bool showSteps) {

    // Resize to fixed 300x300 for consistent grid display
    cv::Mat img;
    cv::resize(src, img, cv::Size(300, 300));
    cv::Mat black = cv::Mat::zeros(img.size(), img.type());

    // ------------------------------------------
    // Step 1: Grayscale, Blur, and Canny (FOR VISUALS ONLY)
    // ------------------------------------------
    cv::Mat gray, blurred, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 50, 150);

    // ------------------------------------------
    // Step 2: Create Color Mask (THE REAL LOGIC)
    // ------------------------------------------
    cv::Mat colorMask = getColorMask(img, colorType);

    // ------------------------------------------
    // Step 3: Find contours on Color Mask
    // ------------------------------------------
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(colorMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Prepare 6 panels
    cv::Mat p1 = img.clone();                    // Original
    cv::Mat p2;                                  // Grayscale
    cv::cvtColor(gray, p2, cv::COLOR_GRAY2BGR);
    cv::Mat p3;                                  // Canny Edges
    cv::cvtColor(edges, p3, cv::COLOR_GRAY2BGR);
    
    cv::Mat p4;                                  // The Color Mask used
    cv::cvtColor(colorMask, p4, cv::COLOR_GRAY2BGR);
    
    cv::Mat p5 = black.clone();                  // Shape (filled green)
    cv::Mat p6 = black.clone();                  // Segmented sign

    std::string shapeName = "No Shape";

    // ------------------------------------------
    // Step 4: Find the largest valid contour
    // ------------------------------------------
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

        if (maxArea > 500) {
            shapeName = classifyShape(contours[largestIdx]);

            // Draw filled green shape with label (p5)
            cv::drawContours(p5, contours, largestIdx, cv::Scalar(0, 255, 0), cv::FILLED);

            // ------------------------------------------
            // Step 5: Segment the sign using the mask
            // ------------------------------------------
            cv::Mat maskGray = cv::Mat::zeros(img.size(), CV_8UC1);
            cv::drawContours(maskGray, contours, largestIdx, cv::Scalar(255), cv::FILLED);
            cv::bitwise_and(img, img, p6, maskGray);
        }
    }

    // ------------------------------------------
    // Step 6: Add labels to each panel
    // ------------------------------------------
    auto addLabel = [](cv::Mat& panel, const std::string& text) {
        int h = panel.rows;
        cv::putText(panel, text, cv::Point(5, h - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    };

    addLabel(p1, "Original");
    addLabel(p2, "Grayscale");
    addLabel(p3, "Canny Edges");
    addLabel(p4, "Color Mask");
    addLabel(p5, shapeName);
    addLabel(p6, "Sign Segmented");

    // ------------------------------------------
    // Step 7: Build 3x2 grid and save
    // ------------------------------------------
    cv::Mat topRow, bottomRow, grid;
    cv::hconcat(std::vector<cv::Mat>{p1, p2, p3}, topRow);
    cv::hconcat(std::vector<cv::Mat>{p4, p5, p6}, bottomRow);
    cv::vconcat(topRow, bottomRow, grid);

    // Save the grid image
    cv::imwrite(outPath, grid);

    // Optionally display
    if (showSteps) {
        std::string winName = "Shape Detection - " + filename;
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 900, 600);
        cv::imshow(winName, grid);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // Print result to console
    std::cout << "  [" << filename << "] Shape: " << shapeName << std::endl;
    return shapeName;
}

// ============================================
// Function: runCameraDemo
// ============================================
// Opens the webcam and performs real-time
// shape and color detection.
// ============================================
void runCameraDemo() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << " LIVE CAMERA MODE ENABLED" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Press 'q' or ESC to quit." << std::endl;
    std::cout << " Press 'm' to toggle split-screen mask view." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    bool showMask = false;
    std::vector<std::string> colors = { "Red Signs", "Blue Signs", "Yellow Signs" };

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Resize for faster processing
        cv::resize(frame, frame, cv::Size(640, 480));
        cv::Mat display = frame.clone();
        cv::Mat allMasks = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Mat allContours = cv::Mat::zeros(frame.size(), CV_8UC3);

        for (const auto& color : colors) {
            cv::Mat mask = getColorMask(frame, color);
            cv::bitwise_or(allMasks, mask, allMasks);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            // Draw all detected contours for the split-screen view
            cv::drawContours(allContours, contours, -1, cv::Scalar(255, 0, 255), 2);

            for (const auto& contour : contours) {
                // 1. Increase minimum area to avoid small background noise
                if (cv::contourArea(contour) > 3000) {
                    
                    // 2. Add Aspect Ratio check (real signs are mostly square-ish proportions)
                    cv::Rect box = cv::boundingRect(contour);
                    float aspectRatio = (float)box.width / (float)box.height;
                    
                    // Most signs (circles, triangles, octagons) have an aspect ratio between 0.6 and 1.4
                    if (aspectRatio > 0.6 && aspectRatio < 1.4) {
                        
                        std::string shape = classifyShape(contour);
                        
                        // 3. Ignore generic polygons (usually background noise)
                        if (shape != "Polygon") { 
                            cv::Scalar boxColor;
                            std::string colorName;
                            if (color == "Red Signs") { boxColor = cv::Scalar(0, 0, 255); colorName = "Red"; }
                            else if (color == "Blue Signs") { boxColor = cv::Scalar(255, 0, 0); colorName = "Blue"; }
                            else { boxColor = cv::Scalar(0, 255, 255); colorName = "Yellow"; }

                            // Draw bounding box and label
                            cv::rectangle(display, box, boxColor, 2);
                            std::string label = colorName + " " + shape;
                            
                            // Add background box for text readability
                            int baseline = 0;
                            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                            cv::rectangle(display, cv::Point(box.x, box.y - textSize.height - 10), 
                                          cv::Point(box.x + textSize.width, box.y), boxColor, cv::FILLED);
                            cv::putText(display, label, cv::Point(box.x, box.y - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                        }
                    }
                }
            }
        }

        if (showMask) {
            cv::Mat p1, p2, p3, p4;
            
            // Resize each panel to half size so the 2x2 grid fits perfectly on screen (320x240 each)
            cv::Size halfSize(frame.cols / 2, frame.rows / 2);
            cv::resize(frame, p1, halfSize);
            
            cv::Mat colorMasks;
            cv::cvtColor(allMasks, colorMasks, cv::COLOR_GRAY2BGR);
            cv::resize(colorMasks, p2, halfSize);
            
            cv::resize(allContours, p3, halfSize);
            cv::resize(display, p4, halfSize);
            
            // Add titles to panels
            cv::putText(p1, "1. Original", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
            cv::putText(p2, "2. HSV Mask", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
            cv::putText(p3, "3. Contours", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
            cv::putText(p4, "4. Final Output", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 2);

            // Construct the 2x2 grid
            cv::Mat topRow, bottomRow, grid;
            cv::hconcat(std::vector<cv::Mat>{p1, p2}, topRow);
            cv::hconcat(std::vector<cv::Mat>{p3, p4}, bottomRow);
            cv::vconcat(topRow, bottomRow, grid);
            
            cv::imshow("Live Shape & Color Detection", grid);
        }
        else {
            cv::imshow("Live Shape & Color Detection", display);
        }

        char key = (char)cv::waitKey(30);
        if (key == 'q' || key == 27) break;
        if (key == 'm') showMask = !showMask;
    }
    cap.release();
    cv::destroyAllWindows();
}

// ============================================
// Main Function
// ============================================
int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << " Member 4: Shape Detection Module" << std::endl;
    std::cout << " MYSignVoice Preliminary Work" << std::endl;
    std::cout << "========================================" << std::endl;

    // Path to the Color Inputs folder
    std::string baseDir = "../../Color Inputs";
    std::string outputDir = "output";
    fs::create_directories(outputDir);

    // Subdirectories containing the 84 test images
    std::vector<std::string> subfolders = { "Red Signs", "Blue Signs", "Yellow Signs" };

    bool showSteps = false;
    
    // Check for arguments
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--camera") {
            runCameraDemo();
            return 0; // Exit after camera mode
        }
        if (arg == "--show") {
            showSteps = true;
            std::cout << "Mode: Step-by-step visualization enabled" << std::endl;
        }
    }

    int totalImages = 0;
    int totalDetected = 0;   // Images where at least one shape was detected
    int totalCircle = 0, totalTriangle = 0, totalRect = 0, totalOctagon = 0, totalPolygon = 0;

    for (const auto& subfolder : subfolders) {
        std::string folderPath = baseDir + "/" + subfolder;
        std::cout << "\nProcessing folder: " << subfolder << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        if (!fs::exists(folderPath)) {
            std::cerr << "  ERROR: Folder not found: " << folderPath << std::endl;
            continue;
        }

        // Create output subfolder
        std::string outSubfolder = outputDir + "/" + subfolder;
        fs::create_directories(outSubfolder);

        int folderTotal = 0, folderDetected = 0;

        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") continue;

            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();

            cv::Mat img = cv::imread(filepath);
            if (img.empty()) {
                std::cerr << "  WARNING: Could not read " << filename << std::endl;
                continue;
            }

            totalImages++;
            folderTotal++;

            // Build output path
            std::string outputPath = outSubfolder + "/Grid_" + filename;

            // Process the image (pass colorType = subfolder name)
            std::string shape = processImage(img, filename, subfolder, outputPath, showSteps);

            if (shape != "No Shape") {
                totalDetected++;
                folderDetected++;

                if (shape == "Circle") totalCircle++;
                else if (shape == "Triangle") totalTriangle++;
                else if (shape == "Rectangle") totalRect++;
                else if (shape == "Octagon") totalOctagon++;
                else if (shape == "Polygon") totalPolygon++;
            }
        }

        // Per-folder statistics
        double folderAcc = (folderTotal > 0) ? (100.0 * folderDetected / folderTotal) : 0;
        std::cout << "  " << subfolder << ": " << folderDetected << "/" << folderTotal
            << " detected (" << std::fixed << std::setprecision(1) << folderAcc << "%)" << std::endl;
    }

    // Overall statistics
    double overallAcc = (totalImages > 0) ? (100.0 * totalDetected / totalImages) : 0;
    std::cout << "\n========================================" << std::endl;
    std::cout << " RESULTS SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Total images processed: " << totalImages << std::endl;
    std::cout << " Total shapes detected:  " << totalDetected << std::endl;
    std::cout << " Overall accuracy:       " << std::fixed << std::setprecision(1) << overallAcc << "%" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << " Shape breakdown:" << std::endl;
    std::cout << "   Circle:    " << totalCircle << std::endl;
    std::cout << "   Triangle:  " << totalTriangle << std::endl;
    std::cout << "   Rectangle: " << totalRect << std::endl;
    std::cout << "   Octagon:   " << totalOctagon << std::endl;
    std::cout << "   Polygon:   " << totalPolygon << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Grid images saved to: " << outputDir << "/" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
