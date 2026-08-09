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
//
// A segmentation counts as SUCCESS only when the extracted region looks like a
// COMPLETE sign: it must sit inside the frame, have no rival contour, and fill
// its own best-fit shape (ellipse for circular signs, rectangle for rectangular
// ones). Partial captures - a crescent or half-disc left behind by glare - are
// reported as FAILED with the reason printed alongside.

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    constexpr double kMinimumContourArea = 500.0;
    constexpr double kMinimumAspectRatio = 0.45;
    constexpr double kMaximumAspectRatio = 2.25;
    constexpr int kPanelSize = 300;

    // --- Success (completeness) thresholds -------------------------------
    // Loose gate used only to keep a contour in the candidate list.
    constexpr double kCandidateSolidity = 0.50;
    // Strict gates applied when deciding SUCCESS vs FAILED.
    // Solidity: how much of its own convex hull the raw contour occupies.
    // A whole disc scores ~0.95 (RETR_EXTERNAL includes the white pictogram);
    // a crescent or C-shape left by a partial mask scores far lower.
    constexpr double kSuccessSolidity = 0.80;
    // Ellipse fill: hull area / best-fit ellipse area. ~1.0 for a full disc,
    // and stays ~1.0 for a disc seen at an angle. A half-disc drops well below.
    constexpr double kMinimumEllipseFill = 0.85;
    // Rect fill: hull area / min-area-rectangle area. ~1.0 for rectangular
    // signs, 0.785 for any circle/ellipse/half-disc - so it only ever rescues
    // genuinely rectangular signs.
    constexpr double kMinimumRectFill = 0.90;

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
        // CLAHE preprocessing to normalize lighting (handles overcast/shaded signs)
        cv::Mat lab;
        cv::cvtColor(source, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> labChannels;
        cv::split(lab, labChannels);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(labChannels[0], labChannels[0]);
        cv::merge(labChannels, lab);
        cv::Mat enhanced;
        cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);

        // GaussianBlur to reduce high-frequency noise before HSV conversion
        cv::Mat blurred;
        cv::GaussianBlur(enhanced, blurred, cv::Size(5, 5), 0);

        cv::Mat hsv;
        cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

        // Blue HSV range matching Member 4's tested values
        // H=[85,135]: captures full range of Malaysian blue signs
        // S=[80,255]: balanced - catches most signs without too much sky noise
        // V=[40,255]: catches shaded/darker signs
        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(85, 80, 40), cv::Scalar(135, 255, 255), mask);

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
            if (aspectRatio < kMinimumAspectRatio || aspectRatio > kMaximumAspectRatio) continue;

            // Solidity check (from shape_detection.cpp): reject noisy/irregular contours.
            // A real sign is solid, not fragmented background noise.
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            const double hullArea = cv::contourArea(hull);
            const double solidity = (hullArea > 0) ? (area / hullArea) : 0;
            if (solidity > kCandidateSolidity) {
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

        std::string reason;
        const auto addReason = [&reason](const std::string& text) {
            if (!reason.empty()) reason += ", ";
            reason += text;
            };

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

            // Use convex hull on the largest contour for cleaner segmentation
            // (from shape_detection.cpp - fixes fragmented contours from white symbols)
            std::vector<cv::Point> hull;
            cv::convexHull(validContours[largestIdx], hull);
            std::vector<std::vector<cv::Point>> hullVec = { hull };

            cv::drawContours(p4, hullVec, 0, cv::Scalar(255, 255, 255), 2);
            cv::drawContours(p5, hullVec, 0, cv::Scalar(255, 255, 255), cv::FILLED);

            cv::Mat filledGray;
            cv::cvtColor(p5, filledGray, cv::COLOR_BGR2GRAY);
            cv::bitwise_and(img, img, p6, filledGray);

            const cv::Rect box = cv::boundingRect(validContours[largestIdx]);
            // Relaxed thresholds: many blue sign photos are close-ups where the
            // sign naturally fills most of the frame or sits near the edge.
            const bool touchesEdge = box.x <= 1 || box.y <= 1 ||
                (box.x + box.width) >= kPanelSize - 1 ||
                (box.y + box.height) >= kPanelSize - 1;
            const bool tooLarge = maxArea > (kPanelSize * kPanelSize * 0.85);

            bool hasCompetitor = false;
            for (size_t i = 0; i < validContours.size(); ++i) {
                if (i == largestIdx) continue;
                if (cv::contourArea(validContours[i]) >= 0.60 * maxArea) {
                    hasCompetitor = true;
                    break;
                }
            }

            // ---- Completeness checks --------------------------------------
            // These are what separate "the whole sign was extracted" from
            // "a chunk of the sign was extracted". Without them a crescent left
            // behind by glare passes as SUCCESS.
            const double hullArea = cv::contourArea(hull);
            const double solidity = (hullArea > 0.0) ? (maxArea / hullArea) : 0.0;

            // Best-fit ellipse: stays near 1.0 for discs viewed head-on OR at an
            // angle, so it tolerates perspective while still rejecting half-discs.
            double ellipseFill = 0.0;
            if (hull.size() >= 5) {
                const cv::RotatedRect fitted = cv::fitEllipse(hull);
                const double ellipseArea = CV_PI * 0.25 *
                    static_cast<double>(fitted.size.width) * fitted.size.height;
                if (ellipseArea > 0.0) ellipseFill = hullArea / ellipseArea;
            }

            // Min-area rectangle: only a genuinely rectangular sign approaches 1.0.
            const cv::RotatedRect minRect = cv::minAreaRect(hull);
            const double minRectArea =
                static_cast<double>(minRect.size.width) * minRect.size.height;
            const double rectFill = (minRectArea > 0.0) ? (hullArea / minRectArea) : 0.0;

            const bool isSolid = solidity >= kSuccessSolidity;
            const bool wholeShape = (ellipseFill >= kMinimumEllipseFill) ||
                (rectFill >= kMinimumRectFill);

            if (touchesEdge)    addReason("touches frame edge");
            if (tooLarge)       addReason("region too large");
            if (hasCompetitor)  addReason("competing contour");
            if (!isSolid)       addReason("fragmented contour");
            if (!wholeShape)    addReason("partial shape");

            success = !touchesEdge && !tooLarge && !hasCompetitor && isSolid && wholeShape;

            if (!success) {
                std::cout << std::fixed << std::setprecision(2)
                    << "    metrics: solidity=" << solidity
                    << " ellipseFill=" << ellipseFill
                    << " rectFill=" << rectFill << '\n';
            }
        }

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
            << (success ? "SUCCESS" : "FAILED");
        if (!success && !reason.empty()) {
            std::cout << " (" << reason << ")";
        }
        std::cout << '\n';
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