// ============================================
// [Member 3] Yellow Sign Segmentation
// Shape-independent edge-assisted version
// ============================================
// Method:
// 1. HSV mask locates likely yellow sign pixels.
// 2. Canny edges locate the closed OUTER sign boundary.
// 3. The best closed edge contour is selected by overlap with the yellow mask.
// 4. The selected outer boundary is filled to segment the whole sign.
// This is not shape classification: it supports any closed sign shape.
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

// Produces a raw HSV mask for display and a gently cleaned mask for contour use.
void getYellowMasks(const cv::Mat& src, cv::Mat& rawMask, cv::Mat& cleanMask) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // Original HSV range used by the baseline program.
    cv::inRange(hsv, cv::Scalar(12, 80, 50),
        cv::Scalar(38, 255, 255), rawMask);

    cleanMask = rawMask.clone();
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(cleanMask, cleanMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(cleanMask, cleanMask, cv::MORPH_CLOSE, kernel);
}

// Finds the best closed edge contour that encloses the largest yellow region.
// An empty vector means that no reliable outer boundary was found.
std::vector<cv::Point> findOuterSignBoundary(const cv::Mat& img,
    const cv::Mat& yellowTargetMask,
    const cv::Point2f& yellowCenter,
    double yellowArea) {
    cv::Mat gray, blurred, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 50, 150);

    // Join tiny breaks in the dark outer sign border.
    cv::Mat edgeKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, edgeKernel);

    std::vector<std::vector<cv::Point>> edgeContours;
    cv::findContours(edges, edgeContours, cv::RETR_LIST,
        cv::CHAIN_APPROX_SIMPLE);

    const double imageArea = static_cast<double>(img.rows * img.cols);
    const double targetPixels = static_cast<double>(cv::countNonZero(yellowTargetMask));
    double bestScore = -1.0;
    std::vector<cv::Point> bestContour;

    for (const auto& contour : edgeContours) {
        const double contourArea = cv::contourArea(contour);
        if (contourArea < 800 || contourArea > imageArea * 0.60) continue;

        // The outside boundary should be at least slightly larger than the
        // yellow region it is expected to enclose.
        if (contourArea < yellowArea * 1.05) continue;

        const cv::Rect box = cv::boundingRect(contour);
        if (box.width < 30 || box.height < 30) continue;

        // Reject very irregular, line-like contours. Traffic-sign outer
        // boundaries are normally compact even when their shape differs.
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        const double hullArea = cv::contourArea(hull);
        if (hullArea <= 0.0) continue;
        const double solidity = contourArea / hullArea;
        if (solidity < 0.60) continue;

        // The yellow-region centre must be inside the possible outer border.
        if (cv::pointPolygonTest(contour, yellowCenter, false) < 0) continue;

        // Measure how much of the largest yellow candidate lies inside this
        // closed edge contour.
        cv::Mat candidateMask = cv::Mat::zeros(img.size(), CV_8U);
        cv::drawContours(candidateMask, std::vector<std::vector<cv::Point>>{contour},
            -1, cv::Scalar(255), cv::FILLED);
        cv::Mat overlapMask;
        cv::bitwise_and(candidateMask, yellowTargetMask, overlapMask);
        const double coverage = cv::countNonZero(overlapMask) /
            std::max(targetPixels, 1.0);
        if (coverage < 0.70) continue;

        // Prefer compact boundaries that contain most of the yellow candidate.
        // The small area term prefers the closest enclosing boundary rather
        // than a very large building or image-border contour.
        const double areaRatio = contourArea / std::max(yellowArea, 1.0);
        const double score = 2.0 * coverage + solidity - 0.03 * areaRatio;

        if (score > bestScore) {
            bestScore = score;
            bestContour = contour;
        }
    }

    return bestContour;
}

// Returns true if a yellow sign candidate is found.
bool processImage(const cv::Mat& src, const std::string& filename,
    const std::string& outPath, bool showSteps) {
    cv::Mat img;
    cv::resize(src, img, cv::Size(300, 300));
    const cv::Mat black = cv::Mat::zeros(img.size(), img.type());

    // 1. Yellow mask.
    cv::Mat rawYellowMask, cleanYellowMask;
    getYellowMasks(img, rawYellowMask, cleanYellowMask);

    // 2. Yellow-region contours.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cleanYellowMask, contours, cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> validContours;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) > 500) {
            validContours.push_back(contour);
        }
    }

    // Required output panels.
    cv::Mat p1 = img.clone();
    cv::Mat p2;
    cv::Mat p3 = black.clone();
    cv::Mat p4 = black.clone();
    cv::Mat p5 = black.clone();
    cv::Mat p6 = black.clone();
    cv::cvtColor(rawYellowMask, p2, cv::COLOR_GRAY2BGR);
    cv::drawContours(p3, validContours, -1, cv::Scalar(255, 0, 255), 2);

    bool detected = false;
    std::string method = "None";

    if (!validContours.empty()) {
        // Locate the largest yellow region. It acts only as a location hint;
        // it is not assumed to be the complete sign boundary.
        int largestIdx = 0;
        double largestYellowArea = 0.0;
        for (int i = 0; i < static_cast<int>(validContours.size()); ++i) {
            const double area = cv::contourArea(validContours[i]);
            if (area > largestYellowArea) {
                largestYellowArea = area;
                largestIdx = i;
            }
        }

        cv::Mat largestYellowMask = cv::Mat::zeros(img.size(), CV_8U);
        cv::drawContours(largestYellowMask, validContours, largestIdx,
            cv::Scalar(255), cv::FILLED);

        const cv::Moments moments = cv::moments(validContours[largestIdx]);
        cv::Point2f yellowCenter;
        if (moments.m00 != 0.0) {
            yellowCenter = cv::Point2f(
                static_cast<float>(moments.m10 / moments.m00),
                static_cast<float>(moments.m01 / moments.m00));
        }
        else {
            const cv::Rect box = cv::boundingRect(validContours[largestIdx]);
            yellowCenter = cv::Point2f(box.x + box.width / 2.0F,
                box.y + box.height / 2.0F);
        }

        // 3. Try to replace the incomplete yellow contour with its complete,
        // shape-independent outer edge boundary.
        const std::vector<cv::Point> outerBoundary = findOuterSignBoundary(
            img, largestYellowMask, yellowCenter, largestYellowArea);

        if (!outerBoundary.empty()) {
            detected = true;
            method = "Outer edge boundary";
            cv::drawContours(p4,
                std::vector<std::vector<cv::Point>>{outerBoundary},
                -1, cv::Scalar(255, 255, 255), 2);
            cv::drawContours(p5,
                std::vector<std::vector<cv::Point>>{outerBoundary},
                -1, cv::Scalar(255, 255, 255), cv::FILLED);
        }
        else {
            // Safe fallback: use the original largest-yellow-contour method.
            detected = true;
            method = "Largest yellow contour (fallback)";
            cv::drawContours(p4, validContours, largestIdx,
                cv::Scalar(255, 255, 255), 2);
            cv::drawContours(p5, validContours, largestIdx,
                cv::Scalar(255, 255, 255), cv::FILLED);
        }

        cv::Mat finalMask;
        cv::cvtColor(p5, finalMask, cv::COLOR_BGR2GRAY);
        cv::bitwise_and(img, img, p6, finalMask);
    }

    auto addLabel = [](cv::Mat& panel, const std::string& text) {
        cv::putText(panel, text, cv::Point(5, panel.rows - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.45,
            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        };

    addLabel(p1, "Original");
    addLabel(p2, "Yellow Mask");
    addLabel(p3, "All Contours");
    addLabel(p4, "Largest Contour");
    addLabel(p5, "Filled Mask");
    addLabel(p6, "Segmented Sign");

    cv::Mat topRow, bottomRow, grid;
    cv::hconcat(std::vector<cv::Mat>{p1, p2, p3}, topRow);
    cv::hconcat(std::vector<cv::Mat>{p4, p5, p6}, bottomRow);
    cv::vconcat(topRow, bottomRow, grid);
    cv::imwrite(outPath, grid);

    if (showSteps) {
        const std::string winName = "Yellow Sign Segmentation - " + filename;
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 900, 600);
        cv::imshow(winName, grid);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "  [" << filename << "] "
        << (detected ? "Yellow Sign Detected - " + method
            : "No Yellow Sign Detected")
        << std::endl;
    return detected;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << " Member 3: Yellow Sign Segmentation" << std::endl;
    std::cout << " MYSignVoice Preliminary Work" << std::endl;
    std::cout << "========================================" << std::endl;

    bool showSteps = false;
#ifdef _DEBUG
    showSteps = true;
    std::cout << "Mode: Debug mode detected. Visualization enabled automatically."
        << std::endl;
#endif
    if (argc > 1 && std::string(argv[1]) == "--show") {
        showSteps = true;
        std::cout << "Mode: Step-by-step visualization enabled" << std::endl;
    }

    std::string baseDir;
    const std::vector<std::string> pathCandidates = {
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
        std::cerr << "ERROR: Could not locate directory 'Color Inputs/Yellow Signs'!"
            << std::endl;
        return -1;
    }

    std::cout << "Reading images from: " << baseDir << std::endl;
    const fs::path outputDir =
        fs::path("preliminary") / "member3_yellow_segmentation" / "output";
    fs::create_directories(outputDir);

    int totalImages = 0;
    int totalDetected = 0;

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
        const std::string outputPath =
            (outputDir / ("Grid_" + filename)).string();
        if (processImage(image, filename, outputPath, showSteps)) {
            ++totalDetected;
        }
    }

    const double detectionRate = totalImages > 0
        ? 100.0 * totalDetected / totalImages : 0.0;

    std::cout << "\n========================================" << std::endl;
    std::cout << " YELLOW SIGN SEGMENTATION SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Total images processed: " << totalImages << std::endl;
    std::cout << " Total yellow signs detected: " << totalDetected << std::endl;
    std::cout << " Detection Rate: " << std::fixed << std::setprecision(1)
        << detectionRate << "%" << std::endl;
    std::cout << " Output folder location: " << outputDir.string() << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
