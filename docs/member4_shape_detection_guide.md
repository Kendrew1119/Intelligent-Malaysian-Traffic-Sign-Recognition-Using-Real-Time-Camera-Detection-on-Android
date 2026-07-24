# Member 4: Shape Detection — Setup, Compile & Run Guide

This guide walks you through setting up OpenCV in Visual Studio 2022, compiling the shape detection code, and running it on the 84 test images.

---

## Prerequisites

Before starting, make sure you have:
1. **Visual Studio 2022** installed (Community Edition is free). During installation, make sure you selected the **"Desktop development with C++"** workload.
2. **OpenCV for Windows** downloaded.

---

## Step 1: Download and Extract OpenCV

1. Go to: https://opencv.org/releases/
2. Download the latest **OpenCV 4.x for Windows** (e.g., `opencv-4.10.0-windows.exe`).
3. Run the `.exe` file. It will ask you where to extract. Extract it to `C:\opencv` so the folder structure looks like:
   ```
   C:\opencv\
     build\
       include\
       x64\
         vc16\
           bin\
           lib\
     sources\
   ```

---

## Step 2: Add OpenCV to System PATH

You need to tell Windows where to find the OpenCV `.dll` files at runtime.

1. Press **Win + R**, type `sysdm.cpl`, press Enter.
2. Go to **Advanced** tab > **Environment Variables**.
3. Under **System variables**, find `Path`, click **Edit**.
4. Click **New** and add: `C:\opencv\build\x64\vc16\bin`
5. Click OK on all windows.
6. **Restart Visual Studio** if it was open.

---

## Step 3: Create a Visual Studio 2022 Project

1. Open Visual Studio 2022.
2. Click **"Create a new project"**.
3. Select **"Console App"** (make sure it says C++ underneath).
4. Project name: `ShapeDetection`
5. Location: `C:\Users\B2B\Desktop\miniproject\preliminary\member4_shape_detection\`
6. Uncheck "Place solution and project in the same directory" if you want a cleaner structure.
7. Click **Create**.

---

## Step 4: Configure OpenCV in the Project

### 4.1: Set Platform to x64
1. In the toolbar at the top, change the platform dropdown from **x86** to **x64**.
2. Change the configuration to **Release** (not Debug, since OpenCV pre-built binaries are Release).

### 4.2: Add Include Directories
1. Right-click your project in Solution Explorer > **Properties**.
2. Make sure **Configuration** is set to **Release** and **Platform** is **x64**.
3. Go to **C/C++** > **General** > **Additional Include Directories**.
4. Add: `C:\opencv\build\include`
5. Click OK.

### 4.3: Add Library Directories
1. Still in Properties, go to **Linker** > **General** > **Additional Library Directories**.
2. Add: `C:\opencv\build\x64\vc16\lib`
3. Click OK.

### 4.4: Add the OpenCV Library File
1. Go to **Linker** > **Input** > **Additional Dependencies**.
2. Add: `opencv_world4100.lib` (replace `4100` with your actual OpenCV version number, e.g., `opencv_world490.lib` for version 4.9.0).
3. Click OK.

### 4.5: Enable C++17 (Required for std::filesystem)
1. Go to **C/C++** > **Language** > **C++ Language Standard**.
2. Set it to **ISO C++17 Standard (/std:c++17)**.
3. Click OK.

---

## Step 5: Add the Source Code

1. In Solution Explorer, find the auto-generated file (e.g., `ShapeDetection.cpp`).
2. **Delete all its contents**.
3. Open the file `shape_detection.cpp` from:
   ```
   c:\Users\B2B\Desktop\miniproject\preliminary\member4_shape_detection\shape_detection.cpp
   ```
4. Copy the entire contents and paste them into your Visual Studio source file.

*(Or you can simply remove the auto-generated .cpp file from the project, then right-click Source Files > Add > Existing Item > select `shape_detection.cpp`)*

---

## Step 6: Set the Working Directory

The code uses relative paths (like `../../Color Inputs`) to find the test images. You need to tell Visual Studio where to run from.

1. Right-click project > **Properties**.
2. Go to **Debugging** > **Working Directory**.
3. Change it to: `$(ProjectDir)` or manually type the full path:
   ```
   c:\Users\B2B\Desktop\miniproject\preliminary\member4_shape_detection\
   ```
4. Click OK.

---

## Step 7: Build and Run

### Build
1. Press **Ctrl + Shift + B** to build the project.
2. If you see errors about `opencv2/opencv.hpp` not found, double-check Step 4.2 (include directories).
3. If you see linker errors about `opencv_world4100.lib`, double-check Step 4.3 and 4.4 (library paths and file name).

### Run
1. Press **Ctrl + F5** (Run without debugging) to execute.
2. The console should print:
   ```
   ========================================
    Member 4: Shape Detection Module
    MYSignVoice Preliminary Work
   ========================================

   Processing folder: Red Signs
   ----------------------------------------
     [000_1_0002.png] Detected 2 shape(s)
     [000_1_0013.png] Detected 1 shape(s)
     ...
   ```
3. The annotated output images are saved to:
   ```
   preliminary/member4_shape_detection/output/Red Signs/
   preliminary/member4_shape_detection/output/Blue Signs/
   preliminary/member4_shape_detection/output/Yellow Signs/
   ```

### Run with Step-by-Step Visualization (Optional)
To see the intermediate images (grayscale, blur, edges) pop up one by one:
1. Go to **Properties** > **Debugging** > **Command Arguments**.
2. Type: `--show`
3. Run again. Each image will show 4 windows. Press any key to move to the next image.

---

## Step 8: Take Screenshots for Report (Chapter 4)

For your Chapter 4 Preliminary Work section, you need to include screenshots of:

1. **Input Image**: The original test image (e.g., `000_1_0002.png`).
2. **Grayscale**: The grayscale conversion.
3. **Canny Edges**: The edge detection output (white lines on black).
4. **Final Result**: The annotated image with bounding boxes and shape labels.

To capture these, run with `--show` mode and use **Win + Shift + S** (Windows Snipping Tool) to screenshot each window.

---

## Step 9: Write Your Report Section

In Chapter 4 of the proposal, Member 4 should write a subsection like this:

```
4.4 Shape Detection — Developed by [Your Name]

4.4.1 Algorithm Description
This module detects the geometric shape of traffic signs using
edge detection and polygon approximation. The algorithm converts
the input image to grayscale, applies Gaussian blur to reduce noise,
then uses Canny edge detection to extract structural boundaries.
The contours are approximated to polygons using cv::approxPolyDP.
Shapes are classified by counting vertices:
- 3 vertices = Triangle (warning signs)
- 4 vertices = Rectangle/Square (regulatory/info signs)
- 7-9 vertices = Octagon (stop signs)
- >9 vertices = Circle (speed limit / prohibitory signs)

4.4.2 Flowchart
[Insert flowchart here]

4.4.3 Results
[Insert screenshots of input, edges, and output for 3-4 sample images]

4.4.4 Parameters and Justification
- Gaussian Blur kernel size: 5x5 (removes noise without blurring sign edges)
- Canny thresholds: 50, 150 (standard ratio of 1:3 for edge detection)
- Polygon approximation epsilon: 3% of perimeter (balances accuracy vs over-simplification)
- Minimum contour area: 800 pixels (filters out small noise while keeping distant signs)
```

---

## Troubleshooting

| Problem | Solution |
| :--- | :--- |
| `cannot open file 'opencv_world4100.lib'` | Check the `.lib` filename matches your OpenCV version (e.g., `opencv_world490.lib` for v4.9.0) |
| `opencv2/opencv.hpp not found` | Re-check the include directory path in Step 4.2 |
| `The program can't find opencv_world4100.dll` | Make sure Step 2 (PATH) was done correctly. Restart VS after changing PATH. |
| `Folder not found: ../../Color Inputs` | Check the Working Directory in Step 6. The code expects to run from inside the `member4_shape_detection` folder. |
| `std::filesystem not found` | Enable C++17 in Step 4.5 |
| Too many false detections | Increase the minimum area threshold from 800 to 1500 in the code |
| Missing real shapes | Decrease the Canny lower threshold from 50 to 30 |
