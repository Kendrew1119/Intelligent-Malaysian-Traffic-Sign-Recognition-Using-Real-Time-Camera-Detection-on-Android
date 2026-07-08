# UCCC2513 Mini Project — Chapter 4: Preliminary Work Coding Guide

This guide explains what the **Chapter 4 (Preliminary Work)** requirement is for your project proposal (Report 1) and provides step-by-step instructions on how each member should implement their C++ OpenCV code.

---

## 1. What is the Preliminary Work Requirement?

In the proposal report (worth 30% of your course marks), the grading lecturer (teacher) wants to see a "proof of concept." You must show that your team can run basic image processing to detect and crop signs from the **84 test images** provided in `Color Inputs`.

To divide the workload among the 4 members, the guideline defines 4 individual modules:
* **Member 1 (Red signs)**: Detect and crop signs with red borders (e.g., Speed limits, Stop, No Entry).
* **Member 2 (Blue signs)**: Detect and crop blue signs (e.g., Mandatory directions, One Way).
* **Member 3 (Yellow signs)**: Detect and crop yellow warning signs (e.g., Warning curve, Junction, School zone).
* **Member 4 (Shape detection)**: Detect shapes (circle, triangle, octagon, rectangle) to verify sign structures.

### "No Name, No Mark" Rule
In Chapter 4 of your report, each member must write a separate subsection tagged with their name (e.g., `4.1 Red Sign Segmentation - Developed by Member Name`). You must include:
1. A description of your C++ algorithm.
2. A flowchart of your code.
3. Screenshots of intermediate steps (HSV mask, contours, final cropped sign bounding boxes) for the test images.

---

## 2. Technical Guide: How Color Segmentation Works (Members 1, 2, & 3)

The standard workflow for traditional color segmentation in OpenCV C++ is:

```
[BGR Input Image] ──► [Convert to HSV] ──► [inRange Thresholding (Mask)]
                                                       │
[Final Bounding Box] ◄── [Find Contours] ◄── [Morphological Cleanup]
```

### Step-by-Step Code Walkthrough

#### Step 2.1: Load Image & Convert to HSV
RGB/BGR is highly sensitive to lighting changes. We convert the image to the **HSV (Hue, Saturation, Value)** color space because Hue isolates the color wavelength, making it independent of brightness.
```cpp
cv::Mat bgr_image = cv::imread("000_1_0002.png");
cv::Mat hsv_image;
cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
```

#### Step 2.2: Threshold for Specific Colors (inRange)
We define the lower and upper bounds of Hue, Saturation, and Value, and create a binary mask (white = target color, black = background).
* **For Red (Member 1)**: Red wraps around the HSV spectrum, so you need two masks and merge them:
  * Mask 1: `H = [0, 10], S = [70, 255], V = [50, 255]`
  * Mask 2: `H = [170, 180], S = [70, 255], V = [50, 255]`
  * `merged_mask = mask1 | mask2;`
* **For Blue (Member 2)**:
  * Range: `H = [100, 130], S = [80, 255], V = [50, 255]`
* **For Yellow (Member 3)**:
  * Range: `H = [15, 35], S = [100, 255], V = [80, 255]`

```cpp
cv::Mat mask;
cv::inRange(hsv_image, cv::Scalar(lowerH, lowerS, lowerV), cv::Scalar(upperH, upperS, upperV), mask);
```

#### Step 2.3: Clean Up Noise (Morphology)
Real-world images contain noise. We use **Morphological Opening** (erosion followed by dilation) to delete small background spots, and **Closing** (dilation followed by erosion) to fill hollow parts inside the sign.
```cpp
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
```

#### Step 2.4: Find Contours & Filter by Geometry
We trace the boundaries of the white regions in the mask. We filter out blobs that are too small or have a non-square aspect ratio (since traffic signs are usually square/circular with aspect ratios close to 1.0).
```cpp
std::vector<std::vector<cv::Point>> contours;
cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area > 500) { // Filter out small noise
        cv::Rect bbox = cv::boundingRect(contour);
        float aspect_ratio = (float)bbox.width / bbox.height;
        if (aspect_ratio >= 0.7 && aspect_ratio <= 1.3) {
            // Draw rectangle on original image
            cv::rectangle(bgr_image, bbox, cv::Scalar(0, 255, 0), 2);
        }
    }
}
```

---

## 3. Technical Guide: How Shape Detection Works (Member 4)

Member 4's task is to analyze the contours of the image and identify their geometric shape to assist the segmentation.

```
[Grayscale Image] ──► [Gaussian Blur] ──► [Canny Edge Detection]
                                                   │
[Polygon Approximation] ◄── [Find Contours] ◄──────┘
```

### Step-by-Step Code Walkthrough

#### Step 3.1: Gray & Blur
Convert the BGR image to grayscale and apply Gaussian blur to smooth out texture noise so we only detect outlines.
```cpp
cv::Mat gray, blurred;
cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
```

#### Step 3.2: Edge Detection
Use Canny edge detection to find the structural boundaries.
```cpp
cv::Mat edges;
cv::Canny(blurred, edges, 50, 150);
```

#### Step 3.3: Polygon Approximation
Find contours on the Canny output. For each contour, use `cv::approxPolyDP` to simplify the boundary curve into a polygon with a limited number of vertices.
```cpp
std::vector<std::vector<cv::Point>> contours;
cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

for (const auto& contour : contours) {
    double perimeter = cv::arcLength(contour, true);
    std::vector<cv::Point> approx;
    // Epsilon is the precision factor (usually 2-4% of perimeter)
    cv::approxPolyDP(contour, approx, 0.03 * perimeter, true);
    
    int vertices = approx.size();
    double area = cv::contourArea(contour);
    
    if (area > 500 && cv::isContourConvex(approx)) {
        if (vertices == 3) {
            // Triangle detected (Yellow warning signs)
        } else if (vertices == 4) {
            // Rectangle/Square detected (Blue signs)
        } else if (vertices == 8) {
            // Octagon detected (Stop signs)
        } else if (vertices > 8) {
            // Circle detected (Speed limits / Red circles)
        }
    }
}
```

---

## 4. Starter Template Files

To get started immediately, go to the folders I created for you. Each folder contains an empty C++ code scaffold with headers and detailed structure comments:

* **Member 1 (Red)**: [preliminary/member1_red_segmentation/red_sign_segmentation.cpp](file:///c:/Users/B2B/Desktop/miniproject/preliminary/member1_red_segmentation/red_sign_segmentation.cpp)
* **Member 2 (Blue)**: [preliminary/member2_blue_segmentation/blue_sign_segmentation.cpp](file:///c:/Users/B2B/Desktop/miniproject/preliminary/member2_blue_segmentation/blue_sign_segmentation.cpp)
* **Member 3 (Yellow)**: [preliminary/member3_yellow_segmentation/yellow_sign_segmentation.cpp](file:///c:/Users/B2B/Desktop/miniproject/preliminary/member3_yellow_segmentation/yellow_sign_segmentation.cpp)
* **Member 4 (Shapes)**: [preliminary/member4_shape_detection/shape_detection.cpp](file:///c:/Users/B2B/Desktop/miniproject/preliminary/member4_shape_detection/shape_detection.cpp)

---

## 5. How to Compile & Run (For Visual Studio Windows Users)

Assuming OpenCV is installed at `C:\opencv`, compile using the Developer Command Prompt for Visual Studio:

1. Open **Developer Command Prompt for VS**.
2. Navigate to your member folder:
   ```bash
   cd c:\Users\B2B\Desktop\miniproject\preliminary\member1_red_segmentation
   ```
3. Compile (replace path to match your local OpenCV version/location):
   ```bash
   cl /EHsc red_sign_segmentation.cpp /I "C:\opencv\build\include" /link /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world460.lib
   ```
4. Run the executable:
   ```bash
   red_sign_segmentation.exe
   ```
