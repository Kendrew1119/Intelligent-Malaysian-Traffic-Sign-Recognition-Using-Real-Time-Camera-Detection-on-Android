// ============================================
// [Member 3] Yellow Sign Segmentation
// Shape-independent edge-assisted version
// ============================================
// Debug update:
// - Each image opens its own debug window.
// - All windows remain open together.
// - Press one key at the end to close all.
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


// Store debug window names
std::vector<std::string> debugWindows;


// Produces raw HSV mask and cleaned mask
void getYellowMasks(
    const cv::Mat& src,
    cv::Mat& rawMask,
    cv::Mat& cleanMask) {


    cv::Mat hsv;

    cv::cvtColor(
        src,
        hsv,
        cv::COLOR_BGR2HSV
    );


    cv::inRange(
        hsv,
        cv::Scalar(12, 80, 50),
        cv::Scalar(38, 255, 255),
        rawMask
    );


    cleanMask = rawMask.clone();


    cv::Mat kernel =
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(3, 3)
        );


    cv::morphologyEx(
        cleanMask,
        cleanMask,
        cv::MORPH_OPEN,
        kernel
    );


    cv::morphologyEx(
        cleanMask,
        cleanMask,
        cv::MORPH_CLOSE,
        kernel
    );
}



// Finds best outer edge boundary
std::vector<cv::Point> findOuterSignBoundary(
    const cv::Mat& img,
    const cv::Mat& yellowTargetMask,
    const cv::Point2f& yellowCenter,
    double yellowArea) {


    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat edges;


    cv::cvtColor(
        img,
        gray,
        cv::COLOR_BGR2GRAY
    );


    cv::GaussianBlur(
        gray,
        blurred,
        cv::Size(5, 5),
        0
    );


    cv::Canny(
        blurred,
        edges,
        50,
        150
    );



    cv::Mat edgeKernel =
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(3, 3)
        );


    cv::morphologyEx(
        edges,
        edges,
        cv::MORPH_CLOSE,
        edgeKernel
    );



    std::vector<std::vector<cv::Point>> edgeContours;


    cv::findContours(
        edges,
        edgeContours,
        cv::RETR_LIST,
        cv::CHAIN_APPROX_SIMPLE
    );



    const double imageArea =
        static_cast<double>(
            img.rows * img.cols
            );


    const double targetPixels =
        static_cast<double>(
            cv::countNonZero(
                yellowTargetMask
            )
            );



    double bestScore = -1.0;


    std::vector<cv::Point> bestContour;



    for (const auto& contour : edgeContours) {


        double contourArea =
            cv::contourArea(contour);



        if (contourArea < 800 ||
            contourArea > imageArea * 0.60)
            continue;



        if (contourArea < yellowArea * 1.05)
            continue;



        cv::Rect box =
            cv::boundingRect(contour);



        if (box.width < 30 ||
            box.height < 30)
            continue;



        std::vector<cv::Point> hull;


        cv::convexHull(
            contour,
            hull
        );



        double hullArea =
            cv::contourArea(hull);



        if (hullArea <= 0)
            continue;



        double solidity =
            contourArea / hullArea;



        if (solidity < 0.60)
            continue;



        if (cv::pointPolygonTest(
            contour,
            yellowCenter,
            false) < 0)
            continue;




        cv::Mat candidateMask =
            cv::Mat::zeros(
                img.size(),
                CV_8U
            );



        cv::drawContours(
            candidateMask,
            std::vector<std::vector<cv::Point>>
        { contour },
            -1,
            cv::Scalar(255),
            cv::FILLED
        );



        cv::Mat overlapMask;



        cv::bitwise_and(
            candidateMask,
            yellowTargetMask,
            overlapMask
        );



        double coverage =
            cv::countNonZero(overlapMask)
            /
            std::max(
                targetPixels,
                1.0
            );



        if (coverage < 0.70)
            continue;



        double areaRatio =
            contourArea /
            std::max(
                yellowArea,
                1.0
            );



        double score =
            2.0 * coverage
            +
            solidity
            -
            0.03 * areaRatio;



        if (score > bestScore) {

            bestScore = score;

            bestContour = contour;
        }
    }



    return bestContour;
}
bool processImage(
    const cv::Mat& src,
    const std::string& filename,
    const std::string& outPath,
    bool showSteps) {


    cv::Mat img;

    cv::resize(
        src,
        img,
        cv::Size(300, 300)
    );


    const cv::Mat black =
        cv::Mat::zeros(
            img.size(),
            img.type()
        );


    // 1. Yellow mask
    cv::Mat rawYellowMask;
    cv::Mat cleanYellowMask;


    getYellowMasks(
        img,
        rawYellowMask,
        cleanYellowMask
    );



    // 2. Find yellow contours
    std::vector<std::vector<cv::Point>> contours;


    cv::findContours(
        cleanYellowMask,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );



    std::vector<std::vector<cv::Point>> validContours;


    for (const auto& contour : contours) {

        if (cv::contourArea(contour) > 500) {

            validContours.push_back(contour);

        }
    }



    cv::Mat p1 = img.clone();

    cv::Mat p2;

    cv::Mat p3 = black.clone();

    cv::Mat p4 = black.clone();

    cv::Mat p5 = black.clone();

    cv::Mat p6 = black.clone();



    cv::cvtColor(
        rawYellowMask,
        p2,
        cv::COLOR_GRAY2BGR
    );



    cv::drawContours(
        p3,
        validContours,
        -1,
        cv::Scalar(255, 0, 255),
        2
    );



    bool detected = false;

    std::string method = "None";



    if (!validContours.empty()) {


        int largestIdx = 0;

        double largestYellowArea = 0;



        for (int i = 0;
            i < static_cast<int>(validContours.size());
            i++) {


            double area =
                cv::contourArea(
                    validContours[i]
                );


            if (area > largestYellowArea) {

                largestYellowArea = area;

                largestIdx = i;
            }
        }



        cv::Mat largestYellowMask =
            cv::Mat::zeros(
                img.size(),
                CV_8U
            );



        cv::drawContours(
            largestYellowMask,
            validContours,
            largestIdx,
            cv::Scalar(255),
            cv::FILLED
        );



        cv::Moments moments =
            cv::moments(
                validContours[largestIdx]
            );



        cv::Point2f yellowCenter;



        if (moments.m00 != 0) {


            yellowCenter =
                cv::Point2f(
                    static_cast<float>(
                        moments.m10 / moments.m00
                        ),
                    static_cast<float>(
                        moments.m01 / moments.m00
                        )
                );

        }
        else {


            cv::Rect box =
                cv::boundingRect(
                    validContours[largestIdx]
                );


            yellowCenter =
                cv::Point2f(
                    box.x + box.width / 2.0F,
                    box.y + box.height / 2.0F
                );
        }




        std::vector<cv::Point> outerBoundary =
            findOuterSignBoundary(
                img,
                largestYellowMask,
                yellowCenter,
                largestYellowArea
            );



        if (!outerBoundary.empty()) {


            detected = true;

            method = "Outer edge boundary";



            cv::drawContours(
                p4,
                std::vector<std::vector<cv::Point>>
            {
                outerBoundary
            },
                -1,
                cv::Scalar(255, 255, 255),
                2
            );



            cv::drawContours(
                p5,
                std::vector<std::vector<cv::Point>>
            {
                outerBoundary
            },
                -1,
                cv::Scalar(255, 255, 255),
                cv::FILLED
            );


        }
        else {


            detected = true;

            method =
                "Largest yellow contour (fallback)";



            cv::drawContours(
                p4,
                validContours,
                largestIdx,
                cv::Scalar(255, 255, 255),
                2
            );



            cv::drawContours(
                p5,
                validContours,
                largestIdx,
                cv::Scalar(255, 255, 255),
                cv::FILLED
            );
        }




        cv::Mat finalMask;


        cv::cvtColor(
            p5,
            finalMask,
            cv::COLOR_BGR2GRAY
        );



        cv::bitwise_and(
            img,
            img,
            p6,
            finalMask
        );

    }




    auto addLabel =
        [](cv::Mat& panel,
            const std::string& text) {


                cv::putText(
                    panel,
                    text,
                    cv::Point(
                        5,
                        panel.rows - 10
                    ),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    cv::Scalar(255, 255, 255),
                    1,
                    cv::LINE_AA
                );
        };



    addLabel(p1, "Original");

    addLabel(p2, "Yellow Mask");

    addLabel(p3, "All Contours");

    addLabel(p4, "Largest Contour");

    addLabel(p5, "Filled Mask");

    addLabel(p6, "Segmented Sign");



    cv::Mat topRow;

    cv::Mat bottomRow;

    cv::Mat grid;



    cv::hconcat(
        std::vector<cv::Mat>
    {
        p1, p2, p3
    },
        topRow
    );



    cv::hconcat(
        std::vector<cv::Mat>
    {
        p4, p5, p6
    },
        bottomRow
    );



    cv::vconcat(
        topRow,
        bottomRow,
        grid
    );



    cv::imwrite(
        outPath,
        grid
    );



    // ===============================
    // DEBUG MODE
    // Open every image separately
    // ===============================
    if (showSteps) {


        std::string winName =
            "Yellow Sign Segmentation - "
            +
            filename;



        cv::namedWindow(
            winName,
            cv::WINDOW_NORMAL
        );



        cv::resizeWindow(
            winName,
            900,
            600
        );



        cv::imshow(
            winName,
            grid
        );



        debugWindows.push_back(
            winName
        );
    }




    std::cout
        << " ["
        << filename
        << "] "
        << (
            detected ?
            "Yellow Sign Detected - "
            + method :
            "No Yellow Sign Detected"
            )
        << std::endl;



    return detected;
}

int main(int argc, char** argv) {


    std::cout
        << "========================================"
        << std::endl;

    std::cout
        << " Member 3: Yellow Sign Segmentation"
        << std::endl;

    std::cout
        << " MYSignVoice Preliminary Work"
        << std::endl;

    std::cout
        << "========================================"
        << std::endl;



    bool showSteps = false;



#ifdef _DEBUG

    showSteps = true;

    std::cout
        << "Mode: Debug mode detected. "
        << "Visualization enabled automatically."
        << std::endl;

#endif



    if (argc > 1 &&
        std::string(argv[1]) == "--show") {

        showSteps = true;

        std::cout
            << "Mode: Step-by-step visualization enabled"
            << std::endl;
    }



    std::string baseDir;



    const std::vector<std::string> pathCandidates =
    {
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


        std::cerr
            << "ERROR: Could not locate directory "
            << "'Color Inputs/Yellow Signs'!"
            << std::endl;


        return -1;
    }



    std::cout
        << "Reading images from: "
        << baseDir
        << std::endl;




    const fs::path outputDir =
        fs::path("preliminary")
        /
        "member3_yellow_segmentation"
        /
        "output";



    fs::create_directories(
        outputDir
    );



    int totalImages = 0;

    int totalDetected = 0;



    for (const auto& entry :
        fs::directory_iterator(baseDir)) {


        if (!entry.is_regular_file())

            continue;



        std::string extension =
            entry.path()
            .extension()
            .string();



        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(
                    std::tolower(c)
                    );
            }
        );



        if (extension != ".png" &&
            extension != ".jpg" &&
            extension != ".jpeg" &&
            extension != ".bmp")

            continue;



        const std::string filename =
            entry.path()
            .filename()
            .string();




        cv::Mat image =
            cv::imread(
                entry.path()
                .string()
            );



        if (image.empty()) {


            std::cerr
                << "WARNING: Failed to read "
                << filename
                << std::endl;


            continue;
        }




        totalImages++;



        const std::string outputPath =
            (
                outputDir /
                ("Grid_" + filename)
                )
            .string();




        if (processImage(
            image,
            filename,
            outputPath,
            showSteps)) {


            totalDetected++;

        }
    }




    double detectionRate =
        totalImages > 0
        ?
        100.0 * totalDetected / totalImages
        :
        0.0;



    std::cout
        << "\n========================================"
        << std::endl;


    std::cout
        << " YELLOW SIGN SEGMENTATION SUMMARY"
        << std::endl;


    std::cout
        << "========================================"
        << std::endl;


    std::cout
        << " Total images processed: "
        << totalImages
        << std::endl;


    std::cout
        << " Total detected: "
        << totalDetected
        << std::endl;


  


    std::cout
        << " Output folder location: "
        << outputDir.string()
        << std::endl;



    std::cout
        << "========================================"
        << std::endl;




    // =====================================
    // Debug mode:
    // Keep all windows open together
    // =====================================
    if (showSteps &&
        !debugWindows.empty()) {


        std::cout
            << "\nAll debug windows opened."
            << std::endl;


        std::cout
            << "Press any key to close all windows..."
            << std::endl;



        cv::waitKey(0);



        cv::destroyAllWindows();
    }



    return 0;
}