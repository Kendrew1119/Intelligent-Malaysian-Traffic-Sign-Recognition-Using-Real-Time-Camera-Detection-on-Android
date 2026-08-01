// ============================================
// [Member 2] blue_sign_segmentation.cpp
// ============================================
// Detects blue Malaysian traffic signs with OpenCV HSV colour segmentation.
//
// Usage:
//   blue_sign_segmentation [image-or-folder]
//
// The argument is optional. If omitted, the program searches a few likely
// relative locations for a "Color Inputs/Blue Signs" folder (see
// findDefaultInputPath), so it can run with Command Arguments left blank in
// Visual Studio as long as Working Directory is set somewhere inside the repo.
//
// Examples:
//   blue_sign_segmentation
//   blue_sign_segmentation "../../Color Inputs/Blue Signs"
//   blue_sign_segmentation "../../Color Inputs/Blue Signs/example.jpg"
//
// Displays a 6-panel grid (Original, Blue Mask, All Contours, Largest
// Contour, Filled Mask, Segmented Sign) live in a window for each image.
// Nothing is written to disk. Press any key to advance to the next image,
// or ESC to quit early.

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    constexpr double kMinimumContourArea = 500.0;
    constexpr double kMinimumAspectRatio = 0.45;
    constexpr double kMaximumAspectRatio = 2.25;
    constexpr int kPanelSize = 300;

    bool isImageFile(const fs::path& path) {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
            extension == ".bmp" || extension == ".tif" || extension == ".tiff";
    }

    // Tries a list of likely folder locations, matching the layout used across
    // the team's preliminary work. Lets the program run with no argument, as
    // long as the working directory is set somewhere inside the repo.
    fs::path findDefaultInputPath() {
        const std::vector<fs::path> candidates = {
            "Color Inputs/Blue Signs",
            "Color Inputs/Traffic signs/Blue Signs",
            "../Color Inputs/Blue Signs",
            "../Color Inputs/Traffic signs/Blue Signs",
            "../../Color Inputs/Blue Signs",
            "../../Color Inputs/Traffic signs/Blue Signs",
        };
        for (const fs::path& candidate : candidates) {
            if (fs::exists(candidate)) {
                return candidate;
            }
        }
        return {};
    }

    std::vector<fs::path> collectInputImages(const fs::path& inputPath) {
        std::vector<fs::path> images;

        if (fs::is_regular_file(inputPath)) {
            if (isImageFile(inputPath)) {
                images.push_back(inputPath);
            }
        }
        else if (fs::is_directory(inputPath)) {
            for (const auto& entry : fs::recursive_directory_iterator(inputPath)) {
                if (entry.is_regular_file() && isImageFile(entry.path())) {
                    images.push_back(entry.path());
                }
            }
        }

        std::sort(images.begin(), images.end());
        return images;
    }

    cv::Mat getBlueMask(const cv::Mat& source) {
        cv::Mat hsv;
        cv::cvtColor(source, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(100, 80, 50), cv::Scalar(130, 255, 255), mask);

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        return mask;
    }

    void addLabel(cv::Mat& panel, const std::string& text) {
        cv::putText(panel, text, cv::Point(5, panel.rows - 10), cv::FONT_HERSHEY_SIMPLEX,
            0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    // Builds the 6-panel grid for one image and returns detection/success flags.
    cv::Mat buildGrid(const cv::Mat& source, const std::string& filename, bool& detected, bool& success) {
        detected = false;
        success = false;

        cv::Mat img;
        cv::resize(source, img, cv::Size(kPanelSize, kPanelSize));
        const cv::Mat black = cv::Mat::zeros(img.size(), img.type());

        const cv::Mat mask = getBlueMask(img);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point>> validContours;
        for (const auto& contour : contours) {
            const double area = cv::contourArea(contour);
            if (area <= kMinimumContourArea) continue;
            const cv::Rect box = cv::boundingRect(contour);
            if (box.width <= 25 || box.height <= 25) continue;
            const double aspectRatio = static_cast<double>(box.width) / box.height;
            if (aspectRatio >= kMinimumAspectRatio && aspectRatio <= kMaximumAspectRatio) {
                validContours.push_back(contour);
            }
        }

        cv::Mat p1 = img.clone();                                     // Original
        cv::Mat p2; cv::cvtColor(mask, p2, cv::COLOR_GRAY2BGR);        // Blue Mask
        cv::Mat p3 = black.clone();                                   // All Contours
        cv::Mat p4 = black.clone();                                   // Largest Contour
        cv::Mat p5 = black.clone();                                   // Filled Mask
        cv::Mat p6 = black.clone();                                   // Segmented Sign

        cv::drawContours(p3, validContours, -1, cv::Scalar(255, 0, 255), 2);

        if (!validContours.empty()) {
            detected = true;

            size_t largestIdx = 0;
            double maxArea = 0.0;
            for (size_t i = 0; i < validContours.size(); ++i) {
                const double area = cv::contourArea(validContours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    largestIdx = i;
                }
            }

            cv::drawContours(p4, validContours, static_cast<int>(largestIdx), cv::Scalar(255, 255, 255), 2);
            cv::drawContours(p5, validContours, static_cast<int>(largestIdx), cv::Scalar(255, 255, 255), cv::FILLED);

            cv::Mat filledGray;
            cv::cvtColor(p5, filledGray, cv::COLOR_BGR2GRAY);
            cv::bitwise_and(img, img, p6, filledGray);

            const cv::Rect box = cv::boundingRect(validContours[largestIdx]);
            const bool touchesEdge = box.x <= 3 || box.y <= 3 ||
                (box.x + box.width) >= kPanelSize - 3 ||
                (box.y + box.height) >= kPanelSize - 3;
            const bool tooLarge = maxArea > (kPanelSize * kPanelSize * 0.45);

            bool hasCompetitor = false;
            for (size_t i = 0; i < validContours.size(); ++i) {
                if (i == largestIdx) continue;
                if (cv::contourArea(validContours[i]) >= 0.30 * maxArea) {
                    hasCompetitor = true;
                    break;
                }
            }

            success = !touchesEdge && !tooLarge && !hasCompetitor;
        }

        const cv::Scalar statusColor = success ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        const std::string statusText = success ? "Segmentation: SUCCESS" : "Segmentation: FAILED";
        cv::putText(p6, statusText, cv::Point(5, 25), cv::FONT_HERSHEY_SIMPLEX, 0.45, statusColor, 1, cv::LINE_AA);

        addLabel(p1, "Original");
        addLabel(p2, "Blue Mask");
        addLabel(p3, "All Contours");
        addLabel(p4, "Largest Contour");
        addLabel(p5, "Filled Mask");
        addLabel(p6, "Segmented Sign");

        cv::Mat topRow, bottomRow, grid;
        cv::hconcat(std::vector<cv::Mat>{p1, p2, p3}, topRow);
        cv::hconcat(std::vector<cv::Mat>{p4, p5, p6}, bottomRow);
        cv::vconcat(topRow, bottomRow, grid);

        std::cout << "[" << filename << "] "
            << (detected ? "Blue Sign Detected - " : "No Blue Sign Detected - ")
            << (success ? "SUCCESS" : "FAILED") << '\n';
        return grid;
    }

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [image-or-folder]\n";
        return 1;
    }

    fs::path inputPath;
    if (argc == 2) {
        inputPath = argv[1];
        if (!fs::exists(inputPath)) {
            std::cerr << "Input path does not exist: " << inputPath << '\n';
            return 1;
        }
    }
    else {
        inputPath = findDefaultInputPath();
        if (inputPath.empty()) {
            std::cerr << "No path given, and could not auto-locate a 'Color Inputs' folder.\n"
                << "Either pass a path as a command-line argument, or set the "
                << "Working Directory (Project Properties -> Debugging) to a folder "
                << "inside the repo that has 'Color Inputs' nearby.\n";
            return 1;
        }
        std::cout << "No argument given, using auto-detected folder: " << inputPath << '\n';
    }

    const std::vector<fs::path> images = collectInputImages(inputPath);
    if (images.empty()) {
        std::cerr << "No supported image files found in: " << inputPath << '\n';
        return 1;
    }

    int totalImages = 0, totalDetected = 0, totalSuccessful = 0;
    const std::string windowName = "Blue Sign Segmentation";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 900, 600);

    for (const fs::path& imagePath : images) {
        const cv::Mat source = cv::imread(imagePath.string());
        if (source.empty()) {
            std::cerr << "Could not read image: " << imagePath << '\n';
            continue;
        }
        ++totalImages;

        bool detected = false, success = false;
        const cv::Mat grid = buildGrid(source, imagePath.filename().string(), detected, success);
        if (detected) ++totalDetected;
        if (success) ++totalSuccessful;

        cv::imshow(windowName, grid);
        const int key = cv::waitKey(0);
        if (key == 27) {  // ESC quits early
            std::cout << "Stopped early by user.\n";
            break;
        }
    }
    cv::destroyAllWindows();

    const double detectionRate = totalImages > 0 ? 100.0 * totalDetected / totalImages : 0.0;
    const double successRate = totalImages > 0 ? 100.0 * totalSuccessful / totalImages : 0.0;

    std::cout << "\n========================================\n"
        << " BLUE SIGN SEGMENTATION SUMMARY\n"
        << "========================================\n"
        << " Total images processed:      " << totalImages << '\n'
        << " Total blue signs detected:    " << totalDetected << '\n'
        << " Total successful segmentations: " << totalSuccessful << '\n'
        << " Detection Rate: " << detectionRate << "%\n"
        << " Success Rate:   " << successRate << "%\n"
        << "========================================\n";

    return 0;
}