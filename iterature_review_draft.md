2.0 Literature Review
2.1 Review on Colour-Based Road Sign Segmentation
Contributed By: Aedan Loh Yi Cheng
2.1.1
Paper: Real-time traffic sign recognition using colour segmentation and SVM
Reference: [3] S. Ardianto, C.-J. Chen, and H.-M. Hang, "Real-time traffic sign recognition using color segmentation and SVM," in Proc. 2017 Int. Conf. Syst., Signals Image Process. (IWSSIP), Poznań, Poland, 2017, pp. 1–5, doi: 10.1109/IWSSIP.2017.7965570.
Summary: 
Ardianto, Chen and Hang (2017) propose a real time traffic sign recognition system that consists of two stages: detection and classification. HSV color segmentation is used for detection, since it is simpler than CIELab (which was found to be more accurate in previous work), and was applied before and after a previous attempt to use histogram equalization (RGB→YCbCr, equalize only the Y channel, then back to HSV) to deal with faded signs. Gaussian/Laplacian filtering is used to remove noise and enhance blurry videos. Candidate boxes are created by Canny edge detection, followed by Suzuki's contour-tracing algorithm, which is filtered by size/shape. HOG and Gabor features are each trained with linear SVMs (HOG: more accurate, slower; Gabor: faster, less accurate) and features with low coefficients are pruned to reduce computation without significantly degrading accuracy. The first several levels of the classification is a hierarchical cascade of binary SVMs (speed limit or not, individual digits), the last level being the rare signs (20/30 km/h). Synthetic data augmentation techniques (blur, lighting changes) lower the accuracy by ~10% compared to real images. Tested on GTSDB, GTSRB and Swedish datasets, the system achieves detection accuracy of 91-94% and classification accuracy of ~98%, like other SVM based systems but somewhat lower than CNN based systems such as Sermanet's.
Block Diagram:
 

Input and Output of Each Block
Block	Input	Output
Raw RGB Image Input	Real-world visual scene captured by a visual sensor.	Raw video frame or image in the RGB color space
Color Space Transformation	Raw RGB image	Image converted into an intensity-independent color space (such as HSV, YUV, CIECAM97, or YCbCr)
Color Channel Isolation	Image in the new, intensity-independent color space	Isolated specific color channels (e.g., Hue channel, U/V channels, or Cr/Cb channels)
Color Thresholding	Isolated color channels	Segmented pixels that match predefined standard traffic sign colors (red, blue, yellow, white, and black)
Detected Traffic Sign Generation	Segmented color pixels	Detected Traffic Sign




2.1.2
Paper: Smart Data Driven Traffic Sign Detection Method Based on Adaptive Colour Threshold and Shape Symmetry
Reference: [1] “Smart data driven traffic sign detection method based on adaptive colour threshold and shape symmetry,” Future Generation Computer Systems, vol. 94, pp. 381–391, May 2019, doi: https://doi.org/10.1016/j.future.2018.11.027.
Summary: 
Under the complex light, Xu et al (2019) detects traffic signs with adaptive color thresholding and shape symmetry testing to improve robustness. To solve the brightness sensitivity of fixed-threshold color methods, images are normalized to Red-Blue grayscale with Red-Blue normalization, then an approximate maximum-minimum normalization step is computed based on the CDF of the gray histogram, where the CDF value is set to 0.9, and the overexposed background is suppressed while the sign foreground is enhanced. Morphological filtering and MSER are used to extract candidate connected regions, and a convex operation corrects contour defects in the occlusion or poor luminosity region. A column-projection feature vector for each region is obtained by Minkowski subtraction to compensate for offset/distortion and then a statistical hypothesis test (t-test/z-test) is performed on the mean and variance of the vector to test for symmetry; symmetric shapes result in near zero values, asymmetric shapes don't. Geometric (size, aspect ratio, area) filters are applied to final ROIs. The method achieves better accuracy, recall, F-measure and execution time than the one proposed by Greenhalgh et al. based on MSER and the one proposed by Garcia-Garrido based on the Hough-transform when tested on GTSDB.
Block Diagram:
 
Input and Output of Each Block
Block	Input	Output
Raw Traffic Scene Image Input	Complex real-world road environment scene	Raw RGB image
Red-Blue Normalization Processing	Raw RGB image	Red-Blue normalized grayscale image
Adaptive Threshold Computation	Red-Blue normalized grayscale image	An adaptive threshold value derived from the Cumulative Distribution Function (CDF) of the image's histogram
Approximate Max-Min Normalization	Red-Blue normalized grayscale image AND the computed adaptive threshold value	Enhanced image with suppressed high-brightness backgrounds and highlighted foreground color blocks.
Detected Traffic Sign Generation	Enhanced and normalized image	Detected Traffic Sign


2.1.3
Paper: Traffic Sign Detection based on Color Segmentation of Obscure Image Candidates: A Comprehensive Study
Reference: Saif, A. F. M. S., Paul, P., Zubair, K. M., Shubho, S. A., & Nandi, D., "Traffic Sign Detection based on Color Segmentation of Obscure Image Candidates: A Comprehensive Study," International Journal of Modern Education and Computer Science (IJMECS), vol. 10, no. 6, pp. 35–46, 6 Jun. 2018, doi: https://doi.org/10.5815/ijmecs.2018.06.05.
Summary:
Nandi et al (2018) discuss TSDR methods including the color segmentation approach, which is one of two approaches for detecting TSDRs. The color detection looks at the five standard sign colors: RGB, but RGB is not a dependable color since the correlated channels of these colors shift towards white/black with different lights and fixed thresholds fail. To overcome this, alternative color spaces are used by researchers: HSV (most common – less lighting sensitive, perceptually similar to human vision) – Paclik et al. thresholded using Hue and Saturation channels for red signs; and Malik et al. thresholded only the Hue channel for red signs; YUV (Shadeed et al.) – thresholded on the U/V channels (U positive, V negative for red); CIECAM97 (Gao et al.) – thresholded using a quad-tree histogram on Hue/chroma; and YCbCr (Dean & Jabir; Chakraborty & Deb) – thresholded on luminance (Y) and chrominance (Cb=blue, Cr=red relative to green). Colour-based methods are faster than shape-based methods but are less accurate in cluttered scenes and are sensitive to illumination changes, motion blur and low light levels, which is why many systems combine colour segmentation with shape analysis (Hough transform, contour approximation, Distance-to-Borders) for increased robustness.


Block Diagram:
 

Input and Output of Each Block
Block	Input	Output
Original RGB Image Input	Real-world visual scene captured by a dashboard camera.	Raw RGB image
YCbCr Conversion & Y-Channel Equalization	Raw RGB image.	Image with clarified/restored luminance (histogram equalization applied exclusively to the Y channel in YCbCr space)
HSV Color Space Transformation	Luminance-equalized YCbCr image	Equalized image translated into the HSV color space.
Fixed Color Range Filtering	Equalized image in the HSV color space.	Segmented binary image where the targeted red spectrum is isolated (marked in white).
Detected Traffic Sign Generation	Segmented binary image	Detected Traffic Sign



2.1.4
Comparison of All Techniques Employed for Colour
Source	Main Technique	Advantage	Disadvantage	Best of Use Context
Paper 1: Real-time traffic sign recognition using colour segmentation and SVM 	Multi-Space Color Segmentation (HSV, YUV, YCbCr) using fixed thresholds.	Color spaces like HSV are less sensitive to lighting variations than RGB; color-based approaches generally produce faster results.	Reduces detection rates in cluttered environments; highly susceptible to physical color changes in reality, such as faded signs.	Favorable lighting conditions (sunny/daylight) and systems where color is used as an initial filter before rigorous shape-based validation.
Paper 2: Smart Data Driven Traffic Sign Detection Method Based on Adaptive Colour Threshold and Shape Symmetry
	Adaptive Thresholding via CDF & Max-Min Normalization on Red-Blue grayscale images.	Actively suppresses interference from high-brightness backgrounds and overexposed areas; highly robust across varying scale, contrast, and illumination.	Leaves behind some noise and isolated points, which requires an additional step of morphological filtering (MSER) to clean the image.	Complex, real-world traffic scenes featuring extreme lighting fluctuations (e.g., heavy glare, deep shadows, low contrast).
Paper 3: Traffic Sign Detection based on Color Segmentation of Obscure Image Candidates: A Comprehensive Study	Histogram Equalization (Y-channel) followed by fixed HSV Color Segmentation.	Effectively restores faded signs and clarifies dark images; computationally efficient, making it highly suitable for real-time processing.	Rigid thresholding can capture unwanted noise, requiring additional Gaussian and Laplacian filters to smooth noise and sharpen motion blur.	Real-time, resource-constrained in-car systems (like dashboard cameras) dealing with moderately blurred or faded video feeds.




 
2.2 Review on Shape Detection & Geometric Analysis for Road Sign
Contributed By: Crystalina Dibble
2.2.1
Title: 
A Hierarchical Approach for Traffic Sign Recognition Based on Shape Detection and Image Classification
References:  
E. H.-C. Lu, M. Gozdzikiewicz, K.-H. Chang, and J.-M. Ciou, "A Hierarchical Approach for Traffic Sign Recognition Based on Shape Detection and Image Classification," Sensors, vol. 22, no. 13, Art. no. 4768, Jun. 2022, doi: 10.3390/s22134768.   
Summary of Technique:
The paper uses shape as an intermediate step to simplify traffic sign recognition. Instead of detecting and classifying all 23 sign types at once, the first stage detects only a sign’s general shape such as circular, triangular, or rectangular since shape broadly indicates a sign’s purpose. For example, circular for prohibitory / mandatory, triangular for warnings. The detected region is then cropped and passed to a second stage for specific classification.
This shape-first split made the detection task much easier R-CNN reached 92.86% mAP when detecting shape alone, versus just 28.64% when detecting and classifying all 23 classes together. The authors conclude that isolating shape as first step is key to making the overall pipeline more accurate and robust, particularly for distant or unclear signs.
Block Diagram:
 
Input and Output of Each Block:
Block	Input	Output
Raw Traffic Signs Image	Camera frame (mobile phone or GoPro)	RGB image
Mask R-CNN (Shape Detection)	RGB image	Bounding polygon/mask and shape category (circular, triangular, or rectangular)
ROI Cropping	Detected shape region and mask coordinates	Cropped single-sign image
Xception (Classification)	Cropped sign image, routed by shape	Class probabilities across the sign's shape-specific categories
Output Sign Label	Class probabilities	Final predicted sign class (1 of 23) and confidence score

Text Description:
Mask R-CNN stage only needs to distinguish 3 shape categories rather than all 23 classes, which is why its output is deliberately limited. The Xception stage then works within a single shape group (e.g., only the 8 circular classes), which is what gives it such high precision (99.73% for circular, 98.45% for triangular) compared to trying to classify all 23 at once.

2.2.2
Title:
Adaptive Perception Driven Traffic Sign Detection using Dynamic ROI Extraction and Hybrid Shape Intelligence
Reference:
A. M. Jenifer and R. Balamanigandan, "Adaptive Perception Driven Traffic Sign Detection using Dynamic ROI Extraction and Hybrid Shape Intelligence," in Proc. 2026 6th Int. Conf. Image Process. Capsule Netw. (ICIPCN), Dhulikhel, Nepal, 2026, pp. 201–205, doi: 10.1109/ICIPCN67432.2026.11438392.   
Summary of Technique:
This paper proposes a hybrid traffic sign detection framework that combines classical image processing with deep learning across three phases. Phase 1 applies preprocessing (median filtering, Gaussian filtering, CLAHE), then fuses GMM-based color segmentation with Canny edge detection, followed by a Hybrid Shape Detection Network (HSD-Net) that classifies candidate regions as circular, triangular, or rectangular using circularity and rectangularity features. Phase 2 refines these candidates through Dynamic ROI Scaling and Edge-Perserving Refinement (DRS-EPR), which adaptively resizes ROIs while preserving sign structure, and extracts HOG features, applies an LSKA attention mechanism, and outputs final classification and bounding box coordinates.
Moreover, over 12,000 images tested, the proposed method achieved 96.3% precision, 94.7% recall, 95.5% F1-score, and 88.9% IoU at 32ms/frame which outperform Color Threshold+Canny, HOG+SVM, and YOLOv3 baselines, as well as CNN and ResNet50 comparisons (97.6% accuracy). The authors conclude that combining classical shape-aware processing with dynamic ROI refinement and lightweight deep learning yields better accuracy and real-time efficiency than end-to-end learning alone, and suggest future work on temporal video information and embedded deployment.
Block Diagram:
 

Input and Output of Each Block:
Block 	Input 	Output
Road Scene Image	Camera frame	Raw RGB image
Preprocessing	Raw RGB image	Denoised, contrast-enhanced image
Color and Edge Segmentation	Enhanced image	Color mask and edge map
Shape detection (HSD-Net)	Color mask and edge map	Shape class (circular/triangular/rectangular) and contour features
Fusion and Candidate ROI	Color, edge, and shape data	Candidate ROI regions
Dynamic ROI Refinement
(DRS-EPR + HOG)	Candidate ROI regions	Refined, scaled ROIs with HOG features
CNN Backbone	Refined ROI image patches	High-dimensional feature maps
Feature Fusion and Attention (LSKA)	CNN feature maps + HOG features	Weighted, fused feature maps
Classification and Bounding Box	Fused feature maps	Class probabilities and bounding box coordinates
Detected Sign and Class (Output)	Class probabilities and bounding box	Final sign label, confidence score, bounding box

2.2.3
Title:
Automatic Traffic Sign Detection and Recognition Using Colour Segmentation and Shape Identification
Reference:
K. Horak, P. Cip, and D. Davidek, "Automatic traffic sign detection and recognition using colour segmentation and shape identification," MATEC Web Conf., vol. 68, p. 17002, 2016, doi: 10.1051/matecconf/20166817002.
Summary of Technique:
This paper proposes a classical (non-deep-learning) two-stage pipeline for detecting and classifying European traffic signs by color and shape. The authors first compare RGB, HSV, and YCbCr color spaces for segmentation and find HSV most effective (97% segmentation efficiency vs. 93% for RGB), since it handles over/under-esposed images better and produces more compact segmentation regions. Candidate regions (ROIs) are etracted by thresholding red, blue, and yellow pixel cluster built from a manually labelled gallery of over 128,000 pixels across 343 images. Each ROI is then normalized to ellipse fitting corrects perspective distortion, rotation is corrected, and the region is trimmed and resampled to 100 x 100 pixel before shape recognition.
For shape classification, the Fast Radical Symmetry (FRS) method detects circular signs by producing a strong grayscale peak whenever a circular shape is present, while the Harris corner detector distinguishes triangle (3 strong corners) from rectangles ( 4 strong corners) based on the number and position of detected corners within the normalized ROI. Tested on real urban driving scenes, the overall system achieved approximately 93% recognition accuracy and processed images fast enough for real-time driver assistance use. The authors note that future work should address signs that visually merge with similarly coloured neighbouring objects, and that better pictogram-recognition method is needed beyond simple correlating matching.

Block Diagram:
 





Input and Output of Each Block:
Block	Input	Output
Input Image	Camera frame (RGB)	Raw RGB image
Colour Space Conversion	RGB image	HSV image
Colour-Based Segmentation	HSV image	Binary maps of red, blue, and yellow candidate regions (ROIs)
Region Normalization	Candidate ROI	Perspective-corrected, rotation-fixed, trimmed 100 x 100 image
Fast Radical Symmetry (Circular Path)	Normalized ROI	Grayscale peak map indicating circular shape presence
Harris Corner Detector (Triangular/Rectangular Path)	Normalized ROI	Detected corner count and positions
Circle Classification	FRS peak map	Confirmed circular sign or rejection
Corner-Based Classification	Corner count and position	Confirmed triangular (3 corners) or rectangular (4 corners) sign
Merge Results	Shape class + colour class	Combined colour-shape candidate label
Detected Sign Class (Output)	Combined candidate label	Final classified traffic sign type

2.3.4 Comparison of All Deep Learning Techniques Employed
Source	Main Technique	Advantanges	Disadvantages	Best Use Context
Paper 1: E. H.-C. Lu, M. Gozdzikiewicz, K.-H. Chang, and J.-M. Ciou, "A Hierarchical Approach for Traffic Sign Recognition Based on Shape Detection and Image Classification," Sensors, vol. 22, no. 13, Art. no. 4768, Jun. 2022, doi: 10.3390/s22134768.   
	Two-stage deep learning: Mask R-CNN detects sign shape (circular/triangular/rectangular), then Xception classifies the specific sign within that shape group	High accuracy (mAP 81.99%, up to 99.7% classification) by letting each model specialize; shape-first split makes detection much easier than direct 23-class detection	Requires large labeled dataset (11,074 images) and two separate models to train/maintain; accuracy drops when train/test devices differ	Applications with access to a large annotated dataset and GPU resources, where maximizing classification accuracy across many sign classes matters most (e.g., autonomous vehicle perception stacks)
Paper 2: A. M. Jenifer and R. Balamanigandan, "Adaptive Perception Driven Traffic Sign Detection using Dynamic ROI Extraction and Hybrid Shape Intelligence," in Proc. 2026 6th Int. Conf. Image Process. Capsule Netw. (ICIPCN), Dhulikhel, Nepal, 2026, pp. 201–205, doi: 10.1109/ICIPCN67432.2026.11438392.   
	Hybrid pipeline: GMM colour segmentation + Canny edge detection feed a Hybrid Shape Detection Network (HSD-Net), refined by Dynamic ROI Scaling (DRS-EPR)	Best overall accuracy/robustness of the three (F1 97%, IoU 88.9%) with fast inference (32ms/frame); combines classical robustness with deep learning precision; handles occlusion and illumination well	Most complex pipeline (7+ stages, multiple hand-tuned modules); heavier computational and implementation overhead than either purely classical or purely deep-learning methods	Real-time ADAS/autonomous driving systems operating in challenging conditions (poor lighting, occlusion, cluttered backgrounds) where both speed and robustness are critical
Paper 3: K. Horak, P. Cip, and D. Davidek, "Automatic traffic sign detection and recognition using colour segmentation and shape identification," MATEC Web Conf., vol. 68, p. 17002, 2016, doi: 10.1051/matecconf/20166817002.
	Classical computer vision: HSV colour segmentation isolates candidate regions, then Fast Radial Symmetry (circles) and Harris corner detection (triangles/rectangles) classify shape	Simple, lightweight, no training data or GPU required; fast and interpretable; easy to implement and tune for a fixed set of geometric shapes	Lower accuracy (~93%) than deep learning approaches; struggles when signs are occluded, faded, or overlap in colour with nearby objects; less robust to real-world variability	Resource-constrained or embedded systems (e.g., early driver-assistance prototypes) where computational simplicity and real-time performance matter more than peak accuracy

 
2.3 Review on Deep Learning for Traffic Sign Recognition
Contributed by: Tan Hui Min
2.3.1 
Title:
Technical evaluation on improved Hough transform for monitoring traffic sign images under depth algorithm 
References:
G. Zhao and J. Ding, “Technical Evaluation on Improved Hough Transform for Monitoring Traffic Sign Images under Depth Algorithm,” International Journal of Dynamical Systems and Differential Equations, vol. 14, no. 2, Jan. 2025, doi: 10.1504/ijdsde.2025.10068898. 
 
Summary of technique
The proposed traffic sign recognition technique combines traditional image processing methods with deep learning to improve recognition accuracy and efficiency. First, traffic sign images captured from road scenes undergo preprocessing, where histogram equalisation is applied to enhance image contrast and bilinear interpolation is used to normalise the images into three different sizes (32×32, 64×64, and 128×128). An improved Hough Transform is then employed to detect traffic sign contours and geometric features using parameter space optimisation, weighted voting based on edge strength, adaptive thresholding, and GPU acceleration, resulting in more accurate and robust feature detection. Next, the three normalised images are processed by separate Convolutional Neural Network (CNN) branches to extract multi-scale features, allowing the model to learn both global and detailed characteristics of traffic signs. The extracted features from all CNN branches are fused into a single feature vector through multi-scale feature fusion. Finally, an Extreme Learning Machine (ELM) classifier classifies the fused features into their corresponding traffic sign categories. Experimental results showed that the proposed hybrid approach achieved a classification accuracy of 99%, outperforming conventional Artificial Neural Network (ANN) and Random Forest methods while also reducing classification time.

System Block Diagram

 
 
Input and Output of Each Block
Block	Input	Output
Image Enhancement	Raw traffic sign image	Enhanced image with improved contrast
Image Size Normalisation	Enhanced image	Three resized images (32×32, 64×64, 128×128)
Improved Hough Transform	Normalised images	Detected traffic sign edges and contours
Multi-scale CNN	Processed images	Feature maps from three CNN branches
Feature Fusion	Three feature maps	Combined feature vector
ELM Classifier	Fused feature vector	Predicted traffic sign class
Recognition Output	Classification result	Final recognised traffic sign
 
Description of Each Block
Block 1: Image Enhancement
The captured traffic sign image is enhanced using histogram equalisation to improve brightness and contrast. This step increases image clarity while preserving colour information, making traffic signs easier to recognise under different lighting conditions. 
Block 2: Image Size Normalisation
The enhanced image is resized into three different resolutions (32×32, 64×64, and 128×128) using bilinear interpolation. Multi-scale images allow the system to recognise traffic signs appearing at different distances and sizes. 
Block 3: Improved Hough Transform
The improved Hough Transform detects traffic sign boundaries by combining parameter space optimisation, weighted voting, adaptive thresholding, and GPU acceleration. Compared with the traditional Hough Transform, this approach improves detection accuracy while reducing computational complexity and false detections. 
Block 4: Multi-scale CNN
Three CNN branches independently process the resized images to extract traffic sign features at different scales. Each CNN consists of convolution layers, ReLU activation, max pooling, and fully connected layers, enabling the model to learn both detailed and global image features. 
Block 5: Multi-scale Feature Fusion
The feature vectors extracted by the three CNN branches are concatenated into a single feature representation. This fusion combines complementary information from different image scales, improving the robustness and recognition performance of the model. 
Block 6: ELM Classifier
The fused feature vector is classified using an Extreme Learning Machine (ELM). Compared with Softmax and SVM classifiers, ELM requires fewer training iterations, reduces computation time, and achieves higher classification accuracy. 
Block 7: Recognition Output
The system outputs the predicted traffic sign category, which can be used in intelligent transportation systems, driver assistance systems, and autonomous vehicles to support safe driving and traffic management.
 
 
 
 


2.3.2 
Title: Deep Learning for Large-ScaleTraffic-Sign Detectionand Recognition
References: 
D. Tabernik and D. Skočaj, "Deep Learning for Large-Scale Traffic-Sign Detection and Recognition," in IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 4, pp. 1427-1440, April 2020, doi: 10.1109/TITS.2019.2913588.

Summary of Techniques
The proposed system employs Mask R-CNN, a deep learning framework for automatic traffic sign detection and recognition. First, road scene images are fed into a Convolutional Neural Network (CNN) backbone, such as ResNet, which extracts hierarchical visual features from the input images. These features are then enhanced by a Feature Pyramid Network (FPN), enabling the model to detect traffic signs of various sizes by combining low-level and high-level feature information. Next, a Region Proposal Network (RPN) generates candidate Regions of Interest (ROIs) that are likely to contain traffic signs. The proposed regions are processed using ROI Align, which accurately extracts fixed-size feature maps while preserving spatial information. Finally, the Mask R-CNN detection heads simultaneously perform three tasks: classifying the traffic sign, refining its bounding box location, and generating a pixel-level segmentation mask. By integrating these components into a single end-to-end trainable framework, the system can accurately detect, localize, classify, and segment traffic signs in complex road environments with high precision and robustness.
 
 













Block diagram of the Studied System
 

Description of Each Block
Block	Input	Output
Image Acquisition	Road image	Digital image
CNN Backbone	Image	Feature maps
Feature Pyramid Network	Feature maps	Multi-scale features
Region Proposal Network	Multi-scale features	Candidate ROIs
ROI Align	ROIs + feature maps	Fixed-size ROI features
Classification Head	ROI features	Traffic sign category
Bounding Box Head	ROI features	Refined bounding box
Mask Head	ROI features	Segmentation mask
Final Output	Detection results	Recognized traffic signs with locations
 
Description of Each Block
Block 1 – Image Acquisition
Road scene images are collected from a traffic-sign dataset and used as the input to the deep learning model.
Block 2 – CNN Backbone
A convolutional neural network extracts hierarchical image features. Early layers capture edges and textures, while deeper layers learn more complex traffic sign patterns.
Block 3 – Feature Pyramid Network (FPN)
The FPN combines features from different CNN layers to detect both small and large traffic signs effectively.
Block 4 – Region Proposal Network (RPN)
The RPN predicts candidate regions that are likely to contain traffic signs, reducing the search space for the detector.
Block 5 – ROI Align
ROI Align extracts accurately aligned feature maps for each proposed region, improving localization accuracy.
Block 6 – Mask R-CNN Heads
The extracted ROI features are processed by three parallel branches:
•	Classification Head predicts the traffic sign category. 
•	Bounding Box Head refines the location of the detected sign. 
•	Mask Head generates a pixel-level segmentation mask for the sign. 
Block 7 – Traffic Sign Detection and Recognition
The outputs from the three branches are combined to produce the results, including the traffic sign category, its bounding box, and segmentation mask.




2.3.3 
Title:
Improved YOLOv8 for small traffic sign detection under complex environmental conditions
References:
B. Ji, J. Xu, Y. Liu, P. Fan, and M. Wang, “Improved YOLOv8 for small traffic sign detection under complex environmental conditions,” Franklin Open, vol. 8, p. 100167, Sep. 2024, doi: 10.1016/j.fraope.2024.100167.

Summarization of Techniques
The proposed system is built upon the YOLOv8n object detection framework and introduces several improvements to enhance the detection of small traffic signs in challenging environments. First, the BoTNet (Bottleneck Transformer) module is incorporated into the backbone network to improve feature extraction by combining convolutional operations with self-attention mechanisms, allowing the model to capture both local and global image information. Next, the ODConv (Omni-dimensional Dynamic Convolution) module is introduced to dynamically adjust convolution kernels according to the input features, enabling the network to focus more effectively on important traffic sign characteristics. The model also integrates the LSKA (Large Separable Kernel Attention) module, which enlarges the receptive field while maintaining low computational complexity, significantly improving the detection of small traffic signs. Furthermore, the conventional IoU loss is replaced with the WIoU (Wise Intersection over Union) loss function to achieve more accurate bounding box regression and stronger generalization capability. By combining these techniques into the YOLOv8 architecture, the proposed model achieves higher detection accuracy and robustness under adverse weather conditions such as fog and low-light environments.
 

Block Diagram of the Studied System

| 
Block	Input	Output
Image Acquisition	Road image	Digital image
Image Preprocessing	Raw image	Normalized input image
YOLOv8 Backbone (BoTNet)	Preprocessed image	Deep feature maps
Feature Extraction	Backbone features	Multi-scale feature maps
Neck (ODConv + LSKA)	Feature maps	Enhanced feature representations
Detection Head	Enhanced features	Traffic sign class probabilities and bounding boxes
WIoU Loss (Training)	Predicted and ground-truth bounding boxes	Optimized model parameters
Final Output	Detection results	Recognized traffic signs with locations and confidence scores
 
Description of Each Block
Block 1 – Image Acquisition
The system receives road scene images captured by a vehicle-mounted camera. These images serve as the input for traffic sign detection.
Block 2 – Image Preprocessing
The input images are resized and normalized to match the input requirements of the YOLOv8 model. This ensures consistent image quality and efficient processing during training and inference.
Block 3 – YOLOv8 Backbone with BoTNet
The preprocessed image is passed through the YOLOv8 backbone, where the BoTNet module enhances feature extraction by combining convolutional operations with transformer-based self-attention. This enables the network to capture both fine details and global contextual information, improving the recognition of traffic signs in complex environments.
Block 4 – Feature Extraction
The backbone produces multi-scale feature maps that represent different levels of image information. These features contain important visual patterns, such as edges, shapes, textures, and semantic information, which are used for detecting traffic signs of various sizes.
Block 5 – Neck Network (ODConv + LSKA)
The extracted feature maps are refined using the neck network, which incorporates the ODConv and LSKA modules. ODConv dynamically adapts convolution kernels according to the input features, improving feature representation, while LSKA enlarges the receptive field and enhances attention to small traffic signs without significantly increasing computational cost.
Block 6 – YOLOv8 Detection Head
The enhanced feature maps are processed by the detection head to simultaneously predict the class of each traffic sign, estimate its confidence score, and determine the coordinates of its bounding box.
Block 7 – WIoU Loss Optimization
During training, the Wise Intersection over Union (WIoU) loss function is used to compare the predicted bounding boxes with the ground-truth annotations. This loss function improves localization accuracy, reduces regression errors, and enhances the model's ability to generalize under different environmental conditions.
Block 8 – Final Traffic Sign Detection
The optimized YOLOv8 model outputs the final detection results, including the location, category, and confidence score for each detected traffic sign. The improved architecture provides higher detection accuracy, particularly for small traffic signs in foggy and low-light conditions.
 




2.3.4 Comparison of All Deep Learning Techniques Employed
Source	Main Technique	Advantage	Disadvantage	Best Use Context
Paper 1: Ding & Zhao (2025), Technical Evaluation on Improved Hough Transform for Monitoring Traffic Sign Images under Depth Algorithm	Hybrid approach combining Improved Hough Transform, Multi-scale CNN, Feature Fusion, and Extreme Learning Machine (ELM) classifier	Combines traditional image processing with deep learning to improve feature extraction and recognition accuracy. Multi-scale CNN captures both local and global features, while ELM provides fast classification with high accuracy.	More complex pipeline due to multiple processing stages. Performance depends on accurate Hough Transform detection before CNN processing, making it less suitable for real-time applications.	Suitable for intelligent transportation systems where high recognition accuracy is more important than processing speed, especially in controlled environments.
Paper 2: Tabernik & Skočaj (2020), Deep Learning for Large-Scale Traffic-Sign Detection and Recognition	Mask R-CNN with ResNet Backbone, Feature Pyramid Network (FPN), Region Proposal Network (RPN), and ROI Align	Provides highly accurate detection, classification, localization, and segmentation simultaneously. FPN improves detection of traffic signs at multiple scales, while ROI Align increases localization precision.	Two-stage detection architecture requires more computational resources and has slower inference speed compared to one-stage detectors, making it less suitable for real-time deployment on low-power devices.	Suitable for large-scale traffic sign inventory, mapping systems, autonomous driving research, and applications requiring high detection accuracy and segmentation.
Paper 3: Ji et al. (2024), Improved YOLOv8 for Small Traffic Sign Detection under Complex Environmental Conditions	YOLOv8n enhanced with BoTNet, ODConv, LSKA, and WIoU Loss	Achieves higher detection accuracy for small traffic signs under foggy and low-light conditions while maintaining real-time detection capability. Attention mechanisms and improved loss function enhance feature representation and localizatioaccuracy.	The additional modules increase model complexity and training time compared with the original YOLOv8. Performance improvements require more computational resources during training.	Best suited for real-time traffic sign detection in autonomous vehicles, Advanced Driver Assistance Systems (ADAS), and intelligent transportation systems operating under challenging environmental conditions.


2.4 Category: CNN Optimization & Robustness in Adverse Conditions
Contributed by: Kendrew Lim Yan Zhe
In the rapid push toward autonomous driving and assistive technologies, deploying real-time object detection on mobile devices presents a massive problem. We want to solve the problem of traffic sign detection for visually impaired walker, but doing this outdoors means our camera system will face severe lighting changes, heavy rain, and motion blur. Traditional colour segmentation fails instantly in these environments. To make our application actually work in real life, we must rely on highly optimized Convolutional Neural Networks (CNNs). This section reviews critical papers that explore how to optimize CNN architectures (like YOLO) for speed and how to make them robust against adverse environmental conditions.

2.4.1: Traffic Sign Detection Under Adverse Environmental Conditions Based on CNN
Reference:
Q. Gao, H. Hu and W. Liu, "Traffic Sign Detection Under Adverse Environmental Conditions Based on CNN," in IEEE Access, vol. 12, pp. 117572-117580, 2024, doi: 10.1109/ACCESS.2024.3446990
Summary of Technique:
This paper addresses the reality that standard traffic sign datasets mostly contain clear, daytime photos. When models trained on these datasets are deployed outdoors, they fail spectacularly during heavy rain, thick fog, or severe sun glare. The core issue is that adverse weather washes out the distinct red and blue chromaticity that neural networks usually rely on to identify signs. 
To counter this, the authors propose a robust CNN pipeline built on two major pillars. First, they use extensive data augmentation to artificially corrupt the training data, forcing the model to learn what a speed limit sign looks like through simulated rain and lens flare. Second, they introduce a Spatial Attention Mechanism into the CNN architecture. Instead of just looking at pixel colours, this attention layer mathematically forces the network to prioritize the geometric structural edges—like the hard outline of an octagon or a triangle—which remain visible even when the paint colour is obscured. This methodology is essential for our project. Since we are building an assistive application for visually impaired walker, the camera will constantly face unpredictable outdoor environments. 
System Block Diagram:
 
Input and Output of Each Block:
Block	Input	Output
Raw Outdoor Image	Camera Frame	RGB Matrix
Data Augmentation Module	Clean RGB Matrix	Distorted image tensors with simulated whether noise (used only in training)
CNN Feature Extractor	Distorted tensor	High-dimensional feature maps
Spatial Attention Layer	Standard Feature Maps	Weighted feature maps highlighting edges and shapes
Classification and Bounding Box	Weighted feature maps	Bounding Box coordinates and class probabilities

Text Description:
The system begins by heavily augmenting the training data to ensure the network is exposed to synthetic adverse weather. The core of the technique lies in the Spatial Attention Layer. Because fog and glare destroy the typical red/blue chromaticity of traffic signs, this layer mathematically forces the CNN to pay more attention to the geometric edges (circles, octagons) which remain visible even when colours fade.
2.4.2 Neural-Network-Based Traffic Sign Detection and Recognition in High-Definition Images Using Region Focusing and Parallelization
Reference:
A. Avramović, D. Sluga, D. Tabernik, D. Skočaj, V. Stojnić and N. Ilc, "Neural-Network-Based Traffic Sign Detection and Recognition in High-Definition Images Using Region Focusing and Parallelization," in IEEE Access, vol. 8, pp. 189855-189868, 2020, doi: 10.1109/ACCESS.2020.3031191

Summary of Techniques:
Processing full high-definition (HD) camera frames through a deep YOLO network is incredibly slow, especially on mobile CPUs. This paper proposes a hybrid architecture approach called "Region Focusing" to solve this hardware limitation.
Instead of feeding the entire high-resolution image into the heavy YOLO network, the system uses fast, parallelized traditional computer vision algorithms (like basic colour thresholding and edge detection) to scan the frame. This lightweight preprocessing step quickly identifies "candidate regions"—small areas that might contain a sign. The system crops these tiny patches (often just 64x64 pixels) and completely discards the remaining 95% of the image containing useless background like sky, trees, and pitch. 
Only these tiny, focused crops are sent to the GPU for the heavy CNN classification. For our mobile deployment, this technique proves that we do not have to compromise between camera resolution and processing speed. By implementing a similar region-focusing pipeline, we can guarantee that our Android app meets the strict real-time performance requirements of the project.
 
System Block Diagram:
 

Input and Output of Each Block:
Block	Input	Output
HD Camera Stream	Live Video Feed	1920x1080 Frames
Parallel Region Focusing	HD Frames	Bounding box coordinates of potential signs.
Candidate Region Crops	HD Frames+ Bounding Box	Small extracted image patches
YOLO/ CNN Classifier	Small Image Patches	Final verified class labels and confidence scores

Text Description:
We noticed that running heavy AI on a phone drains the battery and causes severe lag. This paper fixes that by splitting the work. A fast, traditional algorithm runs on multiple CPU threads to rapidly scan the HD image for anything that looks similarly like a sign. It crops these areas out and discards the rest of the useless background (like trees and sky). The heavy YOLO network then only processes these tiny candidate crops, which keeps the framerate high enough for real-time mobile use.
 
2.4.3 Traffic Sign Detection and Recognition Using YOLO Object Detection Algorithm: A Systematic Review
Reference: 
F. Calero, M. Astudillo, C.G. Bustillos, D. E. Maza, J. Lita, B. Defaz, B. Ante, J. Z. Blanco, D.Armingol, J.M.A. Moreno. (2024). Traffic Sign Detection and Recognition Using YOLO Object Detection Algorithm: A Systematic Review. Mathematics. 12. 297. 10.3390/math12020297. 
Summary of Techniques:
Unlike standard experimental research, this paper is a Systematic Literature Review that tracks the historical evolution of the YOLO architecture from its early v2 iterations all the way to YOLOv8, using public datasets (GTSDB, GTSRB, TT100K), evaluation metrics (mAP, FPS), hardware platforms (NVIDIA GPUs, Jetson Xavier NX, mobile GPU), and challenges in real road conditions. With dozens of object detection architectures available today, selecting the correct model for a mobile edge device is a complex balancing act between Mean Average Precision (accuracy) and Frames Per Second (speed). 
Using the rigorous PRISMA screening framework, the authors analyzed 115 primary studies to extract comparable performance metrics across different hardware setups. The review highlights specific architectural breakthroughs that have made mobile deployment feasible. For instance, it details how the transition to seperate heads and anchor-free detection in YOLOv8 significantly reduces the computational overhead while actually improving the detection of small objects—like distant traffic signs. 
This comprehensive review acts as our academic foundation. Rather than guessing which AI model to use, we can point directly to this paper's statistical metric extraction to justify our architectural choices. 





System Block Diagram:
 
Input and Output of Each Block:
Block	Input	Output
Academic Databases	Search keywords	Raw research paper
PRISMA Screening Protocol	Raw paper	115 filtered, highly relevant primary studies
Architecture Comparison	Selected studies	Structural differences between YOLO generations
Metric Extraction	Experimental data from studies	Statistical Comparison tables of speed vs accuracy
Optimal Edge Strategy	Extracted metrics	Conclusion on the best YOLO variant for mobile deployment

Text Description:
This paper acts as a master guide for selecting the right deep learning architecture. The researchers used the PRISMA framework to systematically filter hundreds of papers down to the most relevant YOLO studies. By extracting and comparing the performance metrics across all these studies, they prove that newer, lightweight YOLO variants (like YOLOv8-nano) offer the only realistic pathway for deploying high-accuracy detection on constrained edge hardware without relying on cloud processing. 
2.4.4 Technical Comparison of 3 Techniques
Feature	Main Technique	Advantage	Disadvantage	Best Use Context
Paper 1: Traffic Sign Detection Under Adverse Environmental Conditions Based on CNN	Synthetic data augmentation and SpatSial Attention Modules.	Drastically reduces false negatives in poor lighting; highly robust for outdoor use.	Training takes significantly longer; attention modules slightly increase inference time.	Deploying the app outdoors in variable weather (rain, fog, nighttime).
Paper 2: Neural-Network-Based Traffic Sign Detection and Recognition in High-Definition Images Using Region Focusing and Parallelization	CPU parallelization of traditional computer vision to crop candidate regions.	Massively speeds up inference time; allows HD camera usage without lagging the phone.	If the fast region focuser misses a faded sign, the CNN never gets a chance to see it.	Processing high-definition real-time mobile camera feeds.
Paper 3: Traffic Sign Detection and Recognition Using YOLO Object Detection Algorithm: A Systematic Review	Systematic PRISMA literature screening and statistical metric extraction.	Provides indisputable, peer-reviewed evidence for selecting our project's YOLOv8 architecture.	Does not propose a new algorithm; relies entirely on the experimental setups of older papers.	Selecting the optimal lightweight AI architecture for edge deployment.

