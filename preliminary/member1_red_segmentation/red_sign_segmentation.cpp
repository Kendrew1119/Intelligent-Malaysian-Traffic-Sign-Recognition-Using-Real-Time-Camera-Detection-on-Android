// ============================================
// [Member 1] red_sign_segmentation.cpp
// ============================================
// Module: Red Sign Segmentation Using Color Information
// Owner: Member 1
//
// Purpose:
//   - Segment RED traffic signs from the 28 test images.
//   - Replicates the Python red_sign_segmentation.py code.
//   - Generates the exact same 6-panel grid output images for Report Chapter 4.
//
// Build (Windows, Visual Studio):
//   cl /EHsc red_sign_segmentation.cpp /I "C:\opencv\build\include" /link /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world460.lib
// ============================================

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void getRedMask(const cv::Mat& src, cv::Mat& mask, cv::Ptr<cv::CLAHE>& clahe) {
    // Apply CLAHE to brighten image in LAB space
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    clahe->apply(channels[0], channels[0]);
    cv::merge(channels, lab);
    
    cv::Mat enhanced;
    cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
    
    // Convert BGR to HSV
    cv::Mat hsv;
    cv::cvtColor(enhanced, hsv, cv::COLOR_BGR2HSV);
    
    // Red has two ranges in HSV (wraps around 0/180)
    cv::Mat mask1, mask2;
    cv::inRange(hsv, cv::Scalar(0, 80, 30), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 80, 30), cv::Scalar(180, 255, 255), mask2);
    
    cv::bitwise_or(mask1, mask2, mask);
    
    // Morphological OPEN and CLOSE using a 5x5 elliptical structuring element
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

bool processImage(const cv::Mat& src, const std::string& filename,
                  const std::string& outPath, cv::Ptr<cv::CLAHE>& clahe,
                  bool& detected, bool showSteps) {
    // Resize to 300x300 for consistent layout
    cv::Mat img;
    cv::resize(src, img, cv::Size(300, 300));
    cv::Mat black = cv::Mat::zeros(img.size(), img.type());
    
    // 1. Get binary mask
    cv::Mat red_mask;
    getRedMask(img, red_mask, clahe);
    
    // 2. Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 3. Contour Filtering (Geometry Check)
    std::vector<std::vector<cv::Point>> valid_contours;
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area > 500) {
            cv::Rect rect = cv::boundingRect(cnt);
            if (rect.width > 25 && rect.height > 25) {
                float aspect_ratio = (float)rect.width / rect.height;
                if (aspect_ratio >= 0.4f && aspect_ratio <= 1.6f) {
                    valid_contours.push_back(cnt);
                }
            }
        }
    }
    
    // Initialize 6 panels
    cv::Mat p1 = img.clone();
    cv::Mat p2;
    cv::cvtColor(red_mask, p2, cv::COLOR_GRAY2BGR);
    cv::Mat p3 = black.clone();
    cv::Mat p4 = black.clone();
    cv::Mat p5 = black.clone();
    cv::Mat p6 = black.clone();
    
    // Draw valid filtered contours in magenta (255, 0, 255) on p3
    cv::drawContours(p3, valid_contours, -1, cv::Scalar(255, 0, 255), 2);
    
    detected = false;
    bool success = false;
    
    // 4. Find largest red contour from valid contours
    if (!valid_contours.empty()) {
        detected = true;
        int largest_idx = 0;
        double max_area = 0;
        for (int i = 0; i < static_cast<int>(valid_contours.size()); ++i) {
            double a = cv::contourArea(valid_contours[i]);
            if (a > max_area) {
                max_area = a;
                largest_idx = i;
            }
        }
        
        if (max_area > 500) {
            // Apply Convex Hull to fix fragmented/broken sign contours
            std::vector<cv::Point> hull;
            cv::convexHull(valid_contours[largest_idx], hull);
            
            // Draw largest outline on p4 in white
            cv::drawContours(p4, std::vector<std::vector<cv::Point>>{hull}, -1, cv::Scalar(255, 255, 255), 2);
            
            // Create filled mask on p5 in white
            cv::drawContours(p5, std::vector<std::vector<cv::Point>>{hull}, -1, cv::Scalar(255, 255, 255), cv::FILLED);
            
            // Segment sign (bitwise AND using p5's filled mask channel 0)
            cv::Mat p5_gray;
            cv::cvtColor(p5, p5_gray, cv::COLOR_BGR2GRAY);
            cv::bitwise_and(img, img, p6, p5_gray);
            
            // 5. Automatic Segmentation Status Heuristics
            // check 1: touches or is within 1 pixel of any image edge (lenient for large/close-up signs)
            bool touches_edge = false;
            cv::Rect rect = cv::boundingRect(hull);
            if (rect.x <= 1 || rect.y <= 1 || (rect.x + rect.width) >= 299 || (rect.y + rect.height) >= 299) {
                touches_edge = true;
            }
            
            // check 2: occupies more than 75% of total image area
            bool too_large = max_area > (90000.0 * 0.75);
            
            // check 3: another valid contour exists whose area is at least 50% of largest
            bool has_competitor = false;
            for (int i = 0; i < static_cast<int>(valid_contours.size()); ++i) {
                if (i == largest_idx) continue;
                double a = cv::contourArea(valid_contours[i]);
                if (a >= 0.50 * max_area) {
                    has_competitor = true;
                    break;
                }
            }
            
            if (!touches_edge && !too_large && !has_competitor) {
                success = true;
            }
        }
    }
    
    // 6. Draw segmentation status label on panel 6
    cv::Scalar status_color = success ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    std::string status_text = success ? "Segmentation: SUCCESS" : "Segmentation: FAILED";
    cv::putText(p6, status_text, cv::Point(5, 25), cv::FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv::LINE_AA);
    
    // 7. Add standard labels
    auto add_label = [](cv::Mat& panel, const std::string& text) {
        cv::putText(panel, text, cv::Point(5, panel.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    };
    
    add_label(p1, "Original");
    add_label(p2, "Red Mask");
    add_label(p3, "All Contours");
    add_label(p4, "Largest Contour");
    add_label(p5, "Filled Mask");
    add_label(p6, "Segmented Sign");
    
    // 8. Stack panels into 3x2 grid
    cv::Mat top_row, bottom_row, grid;
    cv::hconcat(std::vector<cv::Mat>{p1, p2, p3}, top_row);
    cv::hconcat(std::vector<cv::Mat>{p4, p5, p6}, bottom_row);
    cv::vconcat(top_row, bottom_row, grid);
    
    // Save the result
    cv::imwrite(outPath, grid);
    
    // Optional interactive visualization
    if (showSteps) {
        std::string win_name = "Red Sign Segmentation - " + filename;
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(win_name, 900, 600);
        cv::imshow(win_name, grid);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    if (detected) {
        std::cout << "  [" << filename << "] Red Sign Detected - " << (success ? "SUCCESS" : "FAILED") << std::endl;
    } else {
        std::cout << "  [" << filename << "] No Red Sign Detected - FAILED" << std::endl;
    }
    
    return success;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << " Member 1: Red Sign Segmentation (C++)" << std::endl;
    std::cout << " MYSignVoice Preliminary Work" << std::endl;
    std::cout << "========================================" << std::endl;

    bool showSteps = false;
    if (argc > 1 && std::string(argv[1]) == "--show") {
        showSteps = true;
        std::cout << "Mode: Step-by-step visualization enabled" << std::endl;
    }

    std::string baseDir;
    const std::vector<std::string> pathCandidates = {
        "../../Color Inputs/Red Signs",
        "../Color Inputs/Red Signs",
        "Color Inputs/Red Signs"
    };

    for (const auto& candidate : pathCandidates) {
        if (fs::exists(candidate)) {
            baseDir = candidate;
            break;
        }
    }

    if (baseDir.empty()) {
        std::cerr << "ERROR: Could not locate directory 'Color Inputs/Red Signs'!" << std::endl;
        return -1;
    }

    std::cout << "Reading images from: " << baseDir << std::endl;
    
    fs::path exeDir = fs::path(argv[0]).parent_path();
    fs::path outputDir = exeDir / "output";
    fs::create_directories(outputDir);

    // Create CLAHE object
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    int totalImages = 0;
    int totalDetected = 0;
    int totalSuccessful = 0;
    int totalFailed = 0;

    for (const auto& entry : fs::directory_iterator(baseDir)) {
        if (!entry.is_regular_file()) continue;

        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
        if (extension != ".png" && extension != ".jpg" &&
            extension != ".jpeg" && extension != ".bmp") continue;

        const std::string filename = entry.path().filename().string();
        const cv::Mat image = cv::imread(entry.path().string());
        if (image.empty()) {
            std::cerr << "  WARNING: Failed to read image " << filename << std::endl;
            continue;
        }

        ++totalImages;
        const std::string outputPath = (outputDir / ("Grid_" + filename)).string();
        
        bool detected = false;
        bool success = processImage(image, filename, outputPath, clahe, detected, showSteps);
        
        if (detected) {
            ++totalDetected;
        }
        if (success) {
            ++totalSuccessful;
        } else {
            ++totalFailed;
        }
    }

    double detectionRate = totalImages > 0 ? (100.0 * totalDetected / totalImages) : 0.0;
    double successRate = totalImages > 0 ? (100.0 * totalSuccessful / totalImages) : 0.0;

    std::cout << "\n========================================" << std::endl;
    std::cout << " RED SIGN SEGMENTATION SUMMARY (C++)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Total images processed: " << totalImages << std::endl;
    std::cout << " Total red signs detected: " << totalDetected << std::endl;
    std::cout << " Total successful segmentations: " << totalSuccessful << std::endl;
    std::cout << " Total failed segmentations: " << totalFailed << std::endl;
    std::cout << " Detection Rate:         " << std::fixed << std::setprecision(1) << detectionRate << "%" << std::endl;
    std::cout << " Success Rate:           " << std::fixed << std::setprecision(1) << successRate << "%" << std::endl;
    std::cout << " Output folder location: " << outputDir.string() << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
