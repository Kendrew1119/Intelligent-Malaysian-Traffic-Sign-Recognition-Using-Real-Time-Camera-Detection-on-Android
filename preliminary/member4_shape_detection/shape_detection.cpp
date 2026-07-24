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

namespace fs = std::filesystem;

// ============================================
// Function: classifyShape
// ============================================
// Uses CIRCULARITY FIRST to distinguish circles
// from octagons, then falls back to vertex count.
// ============================================
std::string classifyShape(const std::vector<cv::Point>& approx, double area) {
    int vertices = (int)approx.size();
    double perimeter = cv::arcLength(approx, true);
    double circularity = (4.0 * CV_PI * area) / (perimeter * perimeter);

    // ------------------------------------------
    // Priority 1: Check circularity FIRST
    // A perfect circle has circularity = 1.0
    // Circles typically score > 0.75
    // Octagons score around 0.5 - 0.7
    // ------------------------------------------
    if (circularity > 0.75 && vertices >= 6) {
        return "Circle (Speed Limit)";
    }

    // ------------------------------------------
    // Priority 2: Vertex-based classification
    // ------------------------------------------
    if (vertices == 3) {
        return "Triangle (Warning)";
    }
    else if (vertices == 4) {
        cv::Rect bounding = cv::boundingRect(approx);
        float aspectRatio = (float)bounding.width / (float)bounding.height;
        if (aspectRatio >= 0.85 && aspectRatio <= 1.15) {
            return "Square (Regulatory)";
        }
        else {
            return "Rectangle (Info)";
        }
    }
    else if (vertices >= 7 && vertices <= 9 && circularity < 0.75) {
        // Only classify as octagon if circularity is LOW
        // (to prevent circles from being misclassified)
        return "Octagon (Stop)";
    }
    else if (vertices > 9 && circularity > 0.6) {
        return "Circle (Speed Limit)";
    }
    else {
        return "Unknown";
    }
}

// ============================================
// Function: getShapeColor
// ============================================
// Returns a distinct BGR drawing color for each shape.
// ============================================
cv::Scalar getShapeColor(const std::string& shape) {
    if (shape.find("Triangle") != std::string::npos) return cv::Scalar(0, 255, 255);   // Yellow
    if (shape.find("Square") != std::string::npos) return cv::Scalar(255, 0, 0);        // Blue
    if (shape.find("Rectangle") != std::string::npos) return cv::Scalar(255, 128, 0);   // Light Blue
    if (shape.find("Octagon") != std::string::npos) return cv::Scalar(0, 0, 255);       // Red
    if (shape.find("Circle") != std::string::npos) return cv::Scalar(0, 255, 0);        // Green
    return cv::Scalar(200, 200, 200);                                                    // Gray
}

// ============================================
// Helper: Check if two bounding boxes overlap
// significantly (IoU > threshold)
// ============================================
bool boxesOverlap(const cv::Rect& a, const cv::Rect& b, double threshold = 0.3) {
    int interArea = (a & b).area();
    if (interArea == 0) return false;
    int unionArea = a.area() + b.area() - interArea;
    return ((double)interArea / unionArea) > threshold;
}

// ============================================
// Function: processImage
// ============================================
// Main processing pipeline for one image.
// Returns the annotated image with detected shapes.
// ============================================
cv::Mat processImage(const cv::Mat& src, const std::string& filename, bool showSteps) {
    cv::Mat gray, blurred, edges, result;
    result = src.clone();

    // ------------------------------------------
    // Step 1: Convert to Grayscale
    // ------------------------------------------
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // ------------------------------------------
    // Step 2: Apply Gaussian Blur to reduce noise
    // ------------------------------------------
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // ------------------------------------------
    // Step 3: Apply Canny Edge Detection
    // Lower thresholds (30, 100) detect more edges including faded signs
    // ------------------------------------------
    cv::Canny(blurred, edges, 30, 100);

    // ------------------------------------------
    // Step 4: Morphological CLOSE (from lecturer's code)
    // Connects broken edge segments so contours form complete shapes
    // ------------------------------------------
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

    // ------------------------------------------
    // Step 5: Find Contours
    // Use RETR_EXTERNAL to only find OUTER contours
    // This prevents detecting logos/numbers INSIDE signs
    // ------------------------------------------
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Collect all valid detections first, then suppress overlaps
    struct Detection {
        cv::Rect bbox;
        std::string shapeName;
        double area;
        int contourIdx;
    };
    std::vector<Detection> detections;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);

        // Filter: ignore contours that are too small (noise)
        // or too large (entire image border)
        if (area < 500 || area > (src.rows * src.cols * 0.5)) {
            continue;
        }

        // ------------------------------------------
        // Step 6: Approximate the contour to a polygon
        // Use a slightly smaller epsilon (0.02) for more precise
        // vertex count, which helps distinguish circles from octagons
        // ------------------------------------------
        double perimeter = cv::arcLength(contours[i], true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, 0.02 * perimeter, true);

        // ------------------------------------------
        // Step 7: Check aspect ratio (signs are roughly square)
        // ------------------------------------------
        cv::Rect bbox = cv::boundingRect(approx);
        float aspectRatio = (float)bbox.width / (float)bbox.height;
        if (aspectRatio < 0.5 || aspectRatio > 2.0) {
            continue;
        }

        // ------------------------------------------
        // Step 8: Classify the shape
        // ------------------------------------------
        std::string shapeName = classifyShape(approx, area);

        if (shapeName == "Unknown") {
            continue;
        }

        Detection det;
        det.bbox = bbox;
        det.shapeName = shapeName;
        det.area = area;
        det.contourIdx = (int)i;
        detections.push_back(det);
    }

    // ------------------------------------------
    // Step 9: Non-Maximum Suppression (NMS)
    // If multiple boxes overlap, keep only the LARGEST one.
    // This prevents 4-8 detections on the same sign.
    // ------------------------------------------
    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (boxesOverlap(detections[i].bbox, detections[j].bbox, 0.3)) {
                // Suppress the smaller one
                if (detections[i].area >= detections[j].area) {
                    suppressed[j] = true;
                } else {
                    suppressed[i] = true;
                }
            }
        }
    }

    // ------------------------------------------
    // Step 10: Draw only the surviving detections
    // ------------------------------------------
    int shapeCount = 0;
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        shapeCount++;

        cv::Scalar color = getShapeColor(detections[i].shapeName);
        cv::drawContours(result, contours, detections[i].contourIdx, color, 2);
        cv::rectangle(result, detections[i].bbox, color, 2);

        std::string label = detections[i].shapeName + " (" + std::to_string((int)detections[i].area) + "px)";
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point textOrigin(detections[i].bbox.x, detections[i].bbox.y - 5);
        if (textOrigin.y < 15) textOrigin.y = detections[i].bbox.y + detections[i].bbox.height + 15;

        cv::rectangle(result,
            cv::Point(textOrigin.x, textOrigin.y - textSize.height - 4),
            cv::Point(textOrigin.x + textSize.width + 4, textOrigin.y + 4),
            color, cv::FILLED);
        cv::putText(result, label, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // Print summary to console
    std::cout << "  [" << filename << "] Detected " << shapeCount << " shape(s)" << std::endl;

    // Optionally show intermediate steps in a single 2x2 grid window
    if (showSteps) {
        // Convert grayscale/edge images to BGR so they can be merged with the colour result
        cv::Mat grayBGR, blurBGR, edgesBGR;
        cv::cvtColor(gray, grayBGR, cv::COLOR_GRAY2BGR);
        cv::cvtColor(blurred, blurBGR, cv::COLOR_GRAY2BGR);
        cv::cvtColor(edges, edgesBGR, cv::COLOR_GRAY2BGR);

        // Fixed cell size: 400x300 each, total grid = 800x600 (very compact)
        cv::Size cellSize(400, 300);
        cv::Mat g, bl, ed, res;
        cv::resize(grayBGR, g, cellSize);
        cv::resize(blurBGR, bl, cellSize);
        cv::resize(edgesBGR, ed, cellSize);
        cv::resize(result, res, cellSize);

        // Add step labels to each cell
        cv::putText(g, "1. Grayscale", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(bl, "2. Gaussian Blur", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(ed, "3. Canny Edges", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(res, "4. Detected Shapes", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Build 2x2 grid: [gray | blur] on top, [edges | result] on bottom
        cv::Mat topRow, bottomRow, grid;
        cv::hconcat(g, bl, topRow);
        cv::hconcat(ed, res, bottomRow);
        cv::vconcat(topRow, bottomRow, grid);

        std::string winName = "Shape Detection - " + filename;
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 800, 600);
        cv::imshow(winName, grid);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return result;
}

// ============================================
// Main Function
// ============================================
int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << " Member 4: Shape Detection Module" << std::endl;
    std::cout << " MYSignVoice Preliminary Work" << std::endl;
    std::cout << "========================================" << std::endl;

    // Path to the Color Inputs folder (relative to this .cpp location)
    std::string baseDir = "../../Color Inputs";
    std::string outputDir = "output";

    // Create output directory if it does not exist
    fs::create_directories(outputDir);

    // Subdirectories containing the 84 test images
    std::vector<std::string> subfolders = { "Red Signs", "Blue Signs", "Yellow Signs" };

    bool showSteps = false; // Set to true to see intermediate windows

    // Check if user wants to see step-by-step windows
    if (argc > 1 && std::string(argv[1]) == "--show") {
        showSteps = true;
        std::cout << "Mode: Step-by-step visualization enabled" << std::endl;
    }

    int totalImages = 0;
    int totalShapes = 0;

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

        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            // Only process image files
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") continue;

            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();

            // Read the image
            cv::Mat img = cv::imread(filepath);
            if (img.empty()) {
                std::cerr << "  WARNING: Could not read " << filename << std::endl;
                continue;
            }

            totalImages++;

            // Process the image
            cv::Mat result = processImage(img, filename, showSteps);

            // Save the annotated result
            std::string outputPath = outSubfolder + "/" + filename;
            cv::imwrite(outputPath, result);
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << " Done! Processed " << totalImages << " images." << std::endl;
    std::cout << " Results saved to: " << outputDir << "/" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
