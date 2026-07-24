Block Diagram 

Title: Traffic Sign Detection based on Color Segmentation of Obscure Image Candidates: A Comprehensive Study 

Reference: Saif, A. F. M. S., Paul, P., Zubair, K. M., Shubho, S. A., & Nandi, D., "Traffic Sign Detection based on Color Segmentation of Obscure Image Candidates: A Comprehensive Study," International Journal of Modern Education and Computer Science (IJMECS), vol. 10, no. 6, pp. 35–46, 6 Jun. 2018, doi: https://doi.org/10.5815/ijmecs.2018.06.05. 

Original Block Diagram: 

 

Block  

Input  

Output  

Original RGB Image Input 

Real-world visual scene captured by a dashboard camera. 

Raw RGB image 

YCbCr Conversion & Y-Channel Equalization 

Raw RGB image. 

Image with clarified/restored luminance (histogram equalization applied exclusively to the Y channel in YCbCr space) 

HSV Color Space Transformation 

Luminance-equalized YCbCr image 

Equalized image translated into the HSV color space. 

Fixed Color Range Filtering  

Equalized image in the HSV color space.  

Segmented binary image where the targeted red spectrum is isolated (marked in white).  

Detected Traffic Sign Generation  

Segmented binary image  

Detected Traffic Sign  

 

Enhanced / Improved Block Diagram: 

Reference: 

[3] S. Ardianto, C.-J. Chen, and H.-M. Hang, "Real-time traffic sign recognition using color segmentation and SVM," in Proc. 2017 Int. Conf. Syst., Signals Image Process. (IWSSIP), Poznań, Poland, 2017, pp. 1–5, doi: 10.1109/IWSSIP.2017.7965570. 

[1] “Smart data driven traffic sign detection method based on adaptive colour threshold and shape symmetry,” Future Generation Computer Systems, vol. 94, pp. 381–391, May 2019, doi: https://doi.org/10.1016/j.future.2018.11.027.  

Saif, A. F. M. S., Paul, P., Zubair, K. M., Shubho, S. A., & Nandi, D., "Traffic Sign Detection based on Color Segmentation of Obscure Image Candidates: A Comprehensive Study," International Journal of Modern Education and Computer Science (IJMECS), vol. 10, no. 6, pp. 35–46, 6 Jun. 2018, doi: https://doi.org/10.5815/ijmecs.2018.06.05 

 

Summary of Enhanced Diagram: 

Input Image Acquisition: Captures the raw road scene/dashboard video frame in standard RGB space. 

Multi-Color Space Transformation: Converts RGB to robust spaces including YCbCr, HSV, and Red-Blue Normalized Chromaticity. 

Luminance Equalization & Illumination Normalization: Applies Y-channel histogram equalization to clarify faded signs/shadows and Approximate Max-Min Normalization to eliminate overexposure and background highlights. 

Color Channel & Feature Isolation: Extracts target feature channels (Hue, Saturation, and Red-Blue difference channels)  

Hybrid Color Thresholding & Filtering: Merges predefined strict color range windowing (target red ring spectrum) with Adaptive Threshold Computation via CDF  

Detected Traffic Sign Candidates Output: Delivers the segmented traffic sign candidate regions for detection/classification  

 

 

 Chapter 2 – Improved System Block Diagram
For each paper reviewed in Chapter 2, this section presents a second block diagram showing a proposed improved 
design, built on the future-work directions the original authors identified, followed by a description of what was 
changed and why.
Contributed By: Crystalina Dibble
Title:
2.1 A Hierarchical Approach for Traffic Sign Recognition Based on Shape Detection and Image Classification
References:  
E. H.-C. Lu, M. Gozdzikiewicz, K.-H. Chang, and J.-M. Ciou, "A Hierarchical Approach for Traffic Sign 
Recognition Based on Shape Detection and Image Classification," Sensors, vol. 22, no. 13, Art. no. 4768, Jun. 
2022, doi: 10.3390/s22134768.   
Improved Block Diagram:
Text Description of Improved Parts:
The original authors flagged two limitations as future work: the model is too heavy for onboard hardware, and the 
23-class scheme does not cover composite signs that mix pictograms with digits or Chinese characters. The 
improved design addresses both. First, the Mask R-CNN shape-detection stage is swapped for a lightweight variant 
with a MobileNet-style backbone, cutting inference cost so the pipeline is feasible on in-vehicle hardware rather 
than only on a workstation GPU. Second, a fourth shape branch — "Composite / Text region" — is added alongside 
the existing circular, triangular and rectangular branches. Instead of routing this branch to Xception, it feeds an OCR 
/ text-CNN module that reads embedded digits and characters (e.g., speed values, place names), which the original 
per-shape Xception classifiers were never designed to do. A merge step then combines the pictogram class with any 
recognized text into the final label, extending the system beyond pure geometric classification without touching the 
parts of the pipeline that already perform well (99.73% precision on circular signs, 98.45% on triangular).
Title:
2.2 Adaptive Perception Driven Traffic Sign Detection using Dynamic ROI Extraction and Hybrid Shape 
Intelligence
Reference:
M. Jenifer and R. Balamanigandan, "Adaptive Perception Driven Traffic Sign Detection using Dynamic ROI 
Extraction and Hybrid Shape Intelligence," in Proc. 2026 6th Int. Conf. Image Process. Capsule Netw. 
(ICIPCN), Dhulikhel, Nepal, 2026, pp. 201–205, doi: 10.1109/ICIPCN67432.2026.11438392
Improved Block Diagram:
Text Description of Improved Parts:
The authors' stated future work was to add temporal video information and to make the framework deployable on 
embedded systems; the original design only processes single, independent frames and uses a full-size CNN 
backbone. Two changes address this. A "Frame Buffer & Temporal Tracking" stage is inserted right after the video 
input, using optical flow or a Kalman filter across a short window of consecutive frames so a sign detected in one 
frame informs the search in the next — this reduces flicker and false positives caused by momentary occlusion or 
motion blur, which a single-frame pipeline cannot recover from. Second, the CNN backbone used for feature 
extraction is replaced with a lightweight, quantized (pruned, INT8) version, cutting model size and latency so the 
32ms/frame result can be preserved on embedded automotive hardware rather than the "typical workstation 
equipped with a graphics card" the original paper tested on. A temporal-consistency check is added just before final 
classification, cross-checking the current prediction against recent frames to smooth out single-frame 
misclassifications before the bounding box is finalized.
Title:
2.3 Automatic Traffic Sign Detection and Recognition Using Colour Segmentation and Shape Identification
Reference:
K. Horak, P. Cip, and D. Davidek, "Automatic traffic sign detection and recognition using colour segmentation and 
shape identification," MATEC Web Conf., vol. 68, p. 17002, 2016, doi: 10.1051/matecconf/20166817002.
Improved Block Diagram:
Text Description of Improved Parts:
The authors explicitly noted two weaknesses for future work: signs that visually merge with a similarly colored
neighboring object are not separated correctly, and pictogram recognition relies on simple correlation matching, 
which is fragile. The improved design inserts a "Connected-component / watershed separation" stage between colorbased segmentation and region normalization. This splits a single-colored blob into distinct candidate objects before 
the ellipse-fitting and normalization step runs, so a sign next to a similarly-colored car, wall, or object is no longer 
treated as one region. Second, the correlation-based matching implied in the original circle/corner classification 
stage is replaced with a CNN pictogram classifier — a small trained network (in the spirit of the Xception stage used 
in the Lu et al. paper) that recognizes the actual pictogram inside a confirmed circular, triangular, or rectangular 
region, rather than comparing pixel correlation against a fixed template. This should be more robust to lighting 
changes, partial occlusion, and minor rotation than direct correlation, while the proven HSV segmentation, FRS, and 
Harris corner stages (93% overall accuracy) are left unchanged.

Done by Tan Hui Min 

 
Original System 

Scope 

The original system combines the key techniques proposed in the three selected research papers to perform traffic sign detection and recognition. The system begins by preprocessing the captured road image using image enhancement and the Improved Hough Transform to improve image quality and extract traffic sign edges. The processed image is then passed through a deep learning network for feature extraction and multi-scale feature learning. Finally, the YOLOv8 detection head classifies the detected traffic signs and predicts their locations. This combined workflow provides accurate traffic sign recognition under various road environments while integrating the strengths of traditional image processing and deep learning.  

Original Design Concept 

The original system is developed by combining the core techniques from the three reviewed papers. The preprocessing stage adopts histogram equalisation, image normalization, and the Improved Hough Transform from Paper 1 to improve image quality and enhance traffic sign edges. The extracted image is then processed using the YOLOv8 backbone with the BoTNet module from Paper 3 to learn deep image features. To improve the detection of traffic signs with different sizes, the Feature Pyramid Network (FPN) from Paper 2 is used to fuse multi-scale features. The enhanced features are further refined by the ODConv and LSKA modules before being passed to the YOLOv8 detection head for classification and localization. During training, the WIoU loss function is used to optimize bounding box prediction. The final output consists of the detected traffic sign category and its corresponding location. 

 

 

Block 

Input 

Output 

Image Enhancement & Normalization 

Raw road image 

Enhanced image 

Improved Hough Transform 

Enhanced image 

Edge-enhanced image 

YOLOv8 Backbone + BoTNet 

Processed image 

Deep feature maps 

Feature Pyramid Network (FPN) 

Feature maps 

Multi-scale feature maps 

ODConv + LSKA 

Multi-scale features 

Enhanced features 

YOLOv8 Detection Head 

Enhanced features 

Traffic sign category and bounding box 

WIoU Loss 

Predicted and ground-truth bounding boxes 

Optimized model 

Traffic Sign Recognition 

Detection results 

Final recognized traffic sign 

 

Description of Each Block 

Block 1 – Image Enhancement & Normalization 

The captured road image is enhanced using histogram equalisation to improve brightness and contrast. The image is then normalized to a suitable size before feature extraction, allowing the deep learning model to process images consistently. 

Block 2 – Improved Hough Transform 

The Improved Hough Transform detects the edges and geometric shapes of traffic signs while reducing background noise. This step improves the quality of the input features before they are passed into the deep learning network. 

Block 3 – YOLOv8 Backbone with BoTNet 

The processed image is passed through the YOLOv8 backbone integrated with the BoTNet module. This module combines convolution and self-attention mechanisms to extract rich local and global image features. 

Block 4 – Feature Pyramid Network (FPN) 

The Feature Pyramid Network combines feature maps from multiple network layers to produce multi-scale representations. This enables the system to detect traffic signs of different sizes more accurately. 

Block 5 – ODConv + LSKA Module 

The ODConv module dynamically adjusts convolution kernels according to the input image, while the LSKA module enhances attention to small traffic signs by enlarging the receptive field with low computational cost. 

Block 6 – YOLOv8 Detection Head 

The YOLOv8 detection head predicts the traffic sign category, confidence score, and bounding box coordinates simultaneously for each detected traffic sign. 

Block 7 – WIoU Loss 

During training, the WIoU loss function optimizes the predicted bounding boxes against the ground-truth annotations to improve localization accuracy and overall model performance. 

Block 8 – Traffic Sign Recognition 

The final output displays the detected traffic sign together with its class label, confidence score, and bounding box location for use in intelligent transportation systems and autonomous vehicles. 

 

Proposed Enhanced System 

Scope 

The proposed enhanced traffic sign detection and recognition system aims to improve the detection accuracy of small traffic signs under challenging environmental conditions, such as fog, rain, and low-light environments. The enhanced system adopts the original workflow by combining the Improved Hough Transform, Feature Pyramid Network (FPN), and the improved YOLOv8 architecture while introducing two additional preprocessing modules: Adaptive Image Enhancement and Super-Resolution Reconstruction. Adaptive Image Enhancement improves image brightness, contrast, and visibility under adverse weather conditions, while Super-Resolution Reconstruction increases the resolution of small and distant traffic signs before detection. These enhancements enable the deep learning model to extract more discriminative features, resulting in higher recognition accuracy, improved localization, and better overall robustness for intelligent transportation systems and autonomous driving applications. 

 

Enhanced Design Concept 

The enhanced system extends the original design by introducing two additional preprocessing stages before feature extraction. After capturing the road image, the system first performs Adaptive Image Enhancement to automatically adjust brightness, contrast, and visibility according to the environmental conditions. This improves image quality and reduces the impact of fog, shadows, and low illumination. Next, Super-Resolution Reconstruction is applied to increase the resolution of small or distant traffic signs, making their visual features more distinguishable for subsequent detection. The enhanced image is then processed using Image Normalization and the Improved Hough Transform to emphasize traffic sign edges and shapes. Deep feature extraction is performed using the YOLOv8 backbone integrated with the BoTNet module. The extracted features are fused through the Feature Pyramid Network (FPN) and further enhanced using the ODConv and LSKA attention modules to improve small-object representation. Finally, the YOLOv8 detection head predicts the traffic sign category and bounding box, while the WIoU loss function optimizes localization accuracy during training. By integrating these additional enhancement modules, the proposed system is expected to improve the detection of small traffic signs while maintaining the real-time performance of YOLOv8. 

 

 

Input and Output of Each Block 

Block 

Input 

Output 

Adaptive Image Enhancement 

Raw road image 

Enhanced image with improved brightness and contrast 

Super-Resolution Reconstruction 

Enhanced image 

High-resolution image with clearer traffic sign details 

Image Normalization & Resizing 

High-resolution image 

Normalized image 

Improved Hough Transform 

Normalized image 

Edge-enhanced image 

YOLOv8 Backbone + BoTNet 

Processed image 

Deep feature maps 

Feature Pyramid Network (FPN) 

Feature maps 

Multi-scale feature maps 

ODConv + LSKA 

Multi-scale feature maps 

Enhanced feature representation 

YOLOv8 Detection Head 

Enhanced features 

Traffic sign class and bounding box 

WIoU Loss 

Predicted and ground-truth bounding boxes 

Optimized model parameters 

Final Traffic Sign Recognition 

Detection results 

Traffic sign category, confidence score, and bounding box 

 
 

 

Description of Each Block 

Block 1 – Adaptive Image Enhancement 

The input road image first undergoes adaptive image enhancement to automatically improve brightness, contrast, and visibility according to the surrounding environmental conditions. This process reduces the negative effects of fog, shadows, and poor lighting, allowing traffic signs to become more distinguishable before feature extraction. 

Block 2 – Super-Resolution Reconstruction 

The enhanced image is then processed using a super-resolution reconstruction module to increase the resolution of small or distant traffic signs. By restoring fine image details and improving image clarity, this module enables the detection network to recognize small traffic signs more accurately. 

Block 3 – Image Normalization & Resizing 

The high-resolution image is normalized and resized to match the input dimensions required by the YOLOv8 network. This ensures consistent image quality and improves processing efficiency during both training and inference. 

Block 4 – Improved Hough Transform 

The normalized image is processed using the Improved Hough Transform to enhance the edges and geometric structures of traffic signs while suppressing unnecessary background information. This preprocessing step provides clearer visual features for the deep learning network. 

Block 5 – YOLOv8 Backbone with BoTNet 

The processed image is fed into the YOLOv8 backbone integrated with the BoTNet module. This module combines convolutional operations with transformer-based self-attention to capture both local texture information and global contextual features from the image. 

Block 6 – Feature Pyramid Network (FPN) 

The Feature Pyramid Network fuses feature maps from multiple network layers to generate rich multi-scale feature representations. This enables the model to detect traffic signs of different sizes, particularly small signs that appear farther from the camera. 

Block 7 – ODConv + LSKA Attention Module 

The ODConv module dynamically adjusts convolution kernels according to the input features, while the LSKA module expands the receptive field and enhances attention to important traffic sign regions. Together, these modules improve feature representation and increase detection accuracy for small traffic signs. 

Block 8 – YOLOv8 Detection Head 

The enhanced feature maps are processed by the YOLOv8 detection head to simultaneously predict the traffic sign category, confidence score, and bounding box coordinates for each detected object. 

Block 9 – WIoU Loss 

During model training, the Wise Intersection over Union (WIoU) loss function compares the predicted bounding boxes with the ground-truth annotations to optimize localization accuracy and improve the model's generalization performance. 

Block 10 – Final Traffic Sign Recognition 

The system outputs the final traffic sign recognition results, including the detected traffic sign category, confidence score, and bounding box location. These outputs can be used by Advanced Driver Assistance Systems (ADAS), autonomous vehicles, and intelligent transportation systems to support safe and reliable driving. 

 

---

# FINAL GROUP COMBINED BLOCK DIAGRAM

## How This Was Created
This final system design was created by combining the best elements from Member 3 (Tan Hui Min) and Member 4's enhanced block diagrams, then adapting the output stage for our visually impaired pedestrian use case.

| Component | Source |
| :--- | :--- |
| Adaptive Image Enhancement | Member 3's enhanced design |
| Improved Hough Transform | Member 3's original design (Paper 1) |
| YOLOv8 Backbone + BoTNet | Member 3's enhanced design (Paper 3) |
| Feature Pyramid Network (FPN) | Member 3's enhanced design (Paper 2) |
| ODConv + LSKA Attention | Member 3's enhanced design (Paper 3) |
| Region Focusing (Preprocessing) | Member 4's enhanced design (Paper 2) |
| Weather Data Augmentation (Training) | Member 4's enhanced design (Paper 1) |
| Accessibility Output (TTS + Vibration) | Member 4's enhanced design (project goal) |

---

## Combined System Block Diagram

```text
                         ┌──────────────────────────┐
                         │  Android CameraX Feed     │
                         │  (1080p, 30 FPS)          │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  Adaptive Image           │
                         │  Enhancement              │
                         │  (Auto brightness/        │
                         │   contrast correction)    │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  OpenCV Region Focusing   │
                         │  + Hough Transform        │
                         │  (CPU: Filters background,│
                         │   detects sign edges)     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  Candidate Region Crops   │
                         │  (Small focused patches   │
                         │   of potential signs)     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  YOLOv8-nano Backbone     │
                         │  + BoTNet Module          │
                         │  (Deep feature extraction │
                         │   via ncnn on mobile)     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  Feature Pyramid Network  │
                         │  (FPN)                    │
                         │  (Multi-scale detection   │
                         │   for near + far signs)   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  ODConv + LSKA Attention  │
                         │  (Enhances small sign     │
                         │   features, suppresses    │
                         │   background noise)       │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  YOLOv8 Detection Head    │
                         │  (Predicts sign class,    │
                         │   confidence, bounding    │
                         │   box coordinates)        │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  Accessibility Output     │
                         │  Engine                   │
                         │  - Text-to-Speech (TTS)   │
                         │  - Haptic Vibration       │
                         │  - Detection History Log  │
                         └──────────────────────────┘
```

---

## Input and Output of Each Block

| Block | Input | Output |
| :--- | :--- | :--- |
| **Android CameraX Feed** | Real-world environment captured by user's phone | 1920x1080 RGB bitmap frames at 30 FPS |
| **Adaptive Image Enhancement** | Raw HD frames | Brightness/contrast corrected frames (handles fog, rain, nighttime) |
| **OpenCV Region Focusing + Hough Transform** | Enhanced HD frames | Bounding box coordinates of candidate sign regions + edge-enhanced crops |
| **Candidate Region Crops** | Bounding boxes + enhanced frames | Small 64x64 or 128x128 cropped patches (background discarded) |
| **YOLOv8-nano Backbone + BoTNet** | Small image crops | Deep feature maps combining local texture and global context |
| **Feature Pyramid Network (FPN)** | Deep feature maps | Multi-scale feature maps (detects both near and distant signs) |
| **ODConv + LSKA Attention** | Multi-scale feature maps | Refined feature maps with enhanced small-sign representation |
| **YOLOv8 Detection Head** | Refined feature maps | Sign class label, confidence score, bounding box coordinates |
| **Accessibility Output Engine** | Verified sign label + confidence | Spoken audio ("Stop sign ahead"), vibration pattern, history log entry |

---

## Description of the Combined Design

This final group design merges the preprocessing strengths of Member 4's Region Focusing pipeline with the deep learning sophistication of Member 3's enhanced YOLOv8 architecture.

The system begins with Adaptive Image Enhancement to correct for the unpredictable outdoor lighting that a visually impaired pedestrian will encounter. The enhanced frame is then passed through the OpenCV Region Focusing layer, which uses fast CPU-based colour and edge checks (including Hough Transform) to locate candidate sign regions and discard the 95% of the image containing irrelevant background. Only the small, focused crops are sent to the YOLOv8-nano backbone integrated with the BoTNet module, which extracts both local texture and global contextual features. The Feature Pyramid Network fuses multi-scale representations so that both near and distant signs are detected. The ODConv and LSKA attention modules refine the features to improve accuracy on small or partially occluded signs. Finally, the YOLOv8 detection head outputs the class label and bounding box, which are passed to the Accessibility Output Engine for spoken audio feedback and haptic vibration alerts.

During training (in Google Colab), the model uses weather-based data augmentation (simulated rain, fog, glare) and the WIoU loss function to ensure the model generalizes well to real Malaysian road conditions.