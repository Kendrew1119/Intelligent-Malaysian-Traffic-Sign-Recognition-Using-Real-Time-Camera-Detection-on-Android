# CNN Optimization & Robustness in Adverse Conditions

### Overview
In the rapid push toward autonomous driving and assistive technologies, deploying real-time object detection on mobile devices presents a massive hurdle. We want to solve the problem of traffic sign detection for visually impaired pedestrians, but doing this outdoors means our camera system will face severe lighting changes, heavy rain, and motion blur. Traditional colour segmentation fails instantly in these environments. 

To make our application actually work in real life, we must rely on highly optimized Convolutional Neural Networks (CNNs). This section reviews three critical papers that explore how to optimize CNN architectures (like YOLO) for speed and how to make them robust against adverse environmental conditions.

---

### Paper 1: Traffic Sign Detection Under Adverse Environmental Conditions Based on CNN

**Summary of Technique:**
This paper addresses a critical flaw in modern object detection: standard traffic sign datasets (like GTSRB) consist almost entirely of clear, daytime photographs. When models trained on these pristine datasets are deployed outdoors, they fail spectacularly during heavy rain, thick fog, or severe sun glare. The core issue is that adverse weather washes out the distinct red and blue chromaticity that neural networks usually rely on to identify signs.

To counter this, the authors propose a robust CNN pipeline built on two major pillars. First, they use extensive data augmentation to artificially corrupt the training data, forcing the model to learn what a speed limit sign looks like through simulated rain and lens flare. Second, they introduce a Spatial Attention Mechanism into the CNN architecture. Instead of just looking at pixel colours, this attention layer mathematically forces the network to prioritize the geometric structural edges—like the hard outline of an octagon or a triangle—which remain visible even when the paint colour is obscured.

This methodology is essential for our project. Since we are building an assistive application for visually impaired pedestrians, the camera will constantly face unpredictable outdoor environments. By adopting this paper's data augmentation strategy, we ensure our model remains highly accurate and reliable regardless of the Malaysian weather.

**System Block Diagram:**
```text
[Raw Outdoor Image] 
       │
       ▼
[Data Augmentation Module] ──► (Simulates Rain/Fog/Glare during training)
       │
       ▼
[CNN Feature Extractor] ──► (Extracts deep image features)
       │
       ▼
[Spatial Attention Layer] ──► (Weights structural features higher than colour)
       │
       ▼
[Classification & BBox]
       │
       ▼
[Final Sign Prediction]
```

**Input and Output of Each Block:**
*   **Raw Outdoor Image:** Input: Camera frame. Output: RGB matrix. *(Image Recommendation: Photo of a traffic sign in heavy rain or thick fog)*
*   **Data Augmentation Module:** Input: Clean RGB matrix. Output: Distorted image tensors with simulated weather noise (used only in training).
*   **CNN Feature Extractor:** Input: Distorted tensors. Output: High-dimensional feature maps. *(Image Recommendation: Crop of the Encoder section from Figure 2 of the paper)*
*   **Spatial Attention Layer:** Input: Standard feature maps. Output: Weighted feature maps highlighting edges and shapes.
*   **Classification & BBox:** Input: Weighted feature maps. Output: Bounding box coordinates and class probabilities.

**Text Description:**
The system begins by heavily augmenting the training data to ensure the network is exposed to synthetic adverse weather. The core of the technique lies in the Spatial Attention Layer. Because fog and glare destroy the typical red/blue chromaticity of traffic signs, this layer mathematically forces the CNN to pay more attention to the geometric edges (circles, octagons) which remain visible even when colours fade.

**Strengths:**
1. **Massive speed improvement in adverse conditions.** The proposed pipeline achieves approximately 5× higher FPS than the previous segmentation-based state-of-the-art (12.79 FPS vs. 2.76 FPS on average across rain, snow and fog), making it the only reviewed method fast enough for real-time outdoor deployment.
2. **Effective data augmentation strategy.** By synthetically generating rain, fog and glare during training, the model generalises to unseen weather conditions without requiring expensive real-world adverse-weather datasets.
3. **Attention mechanism preserves geometric cues.** The Spatial Attention Layer shifts the network's focus from colour (which fades in bad weather) to structural edges (circles, triangles, octagons), which remain visible even under heavy fog or sun glare.

**Weaknesses:**
1. **Slightly lower absolute accuracy.** The method trades a small accuracy decrease (approximately −2.8% average across conditions) compared to heavier segmentation models such as SegU-Net, meaning it is not the most precise detector when speed is not a concern.
2. **Synthetic augmentation may not fully represent real weather.** The simulated rain and fog patterns are algorithmically generated and may not capture every real-world degradation pattern (e.g., water droplets on the lens, windshield reflections).
3. **Limited evaluation on non-European sign sets.** The experiments use primarily European and Chinese traffic sign datasets; performance on Malaysian signs with different colour palettes and pictograms is not directly validated.

---

### Paper 2: Neural-Network-Based Traffic Sign Detection and Recognition in High-Definition Images Using Region Focusing and Parallelization

**Summary of Technique:**
The study evaluates region focusing on approximately full-HD input (about 1920×1054 pixels). Instead of classifying arbitrary tiny patches, it forms three relatively large fixed or movable regions of interest (ROIs), with network-input sizes in the 608×608 to 704×704 range. The YOLO detector is retrained with ROI-derived images so that its training input matches the focused deployment input. The ROI branches can then be processed in parallel.

The reported results must be interpreted together with the hardware and dataset configuration. The approximately 17 FPS result used three GPUs, whereas the reported single-GPU movable-ROI results were about 7–8 FPS. The higher 93.71% mAP result also used an extended training dataset, so the gain cannot be attributed to cropping alone. In addition, the paper's optimized custom CUDA preprocessing was substantially faster than its general OpenCV implementation. The study therefore supports benchmarking region focusing for high-resolution scenes, but it does not guarantee a speed or accuracy improvement on a single laptop CPU/GPU.

**System Block Diagram:**
```text
[Full-HD Image (approximately 1920×1054)]
       │
       ▼
[Fixed / Movable Region Focusing]
       │
       ▼
[Three Large ROI Images] ──► (approximately 608×608 to 704×704)
       │
       ▼
[ROI-Trained YOLO Detectors] ──► (parallel GPU processing in the fastest setup)
       │
       ▼
[Detected Traffic Signs]
```

**Input and Output of Each Block:**
*   **Full-HD Image:** Input: High-definition road image. Output: Approximately 1920×1054 pixels for the reported experiment. *(Image Recommendation: A wide road-scene image in which signs occupy a small part of the frame)*
*   **Fixed / Movable Region Focusing:** Input: Full image. Output: Coordinates for three large search regions selected according to the paper's focusing strategy.
*   **Large ROI Images:** Input: Full image + ROI coordinates. Output: ROI inputs of approximately 608×608 to 704×704 pixels with surrounding road-scene context. *(Image Recommendation: The three overlapping search regions illustrated by the paper)*
*   **ROI-Trained YOLO Detectors:** Input: Large ROI images. Output: detected boxes, classes and confidence scores. The detector is retrained using corresponding ROI-derived training data.

**Text Description:**
The paper divides a high-definition image into focused search regions and can process those ROI branches in parallel. Because the regions are still hundreds of pixels wide, the approach retains useful context and makes small signs larger relative to each network input. Its benefit depends on ROI implementation, detector retraining and available hardware. It should therefore be treated as an experimentally testable optimization, not as proof that any colour/contour crop will automatically be faster or more accurate.

**Strengths:**
1. **Preserves detail for small signs in high-resolution scenes.** Large focused inputs make signs occupy a greater proportion of the detector input while retaining surrounding context.
2. **Parallelisation-friendly architecture.** The region-focusing design creates independent ROI branches that can be distributed across multiple GPUs; this is the context for the paper's fastest result.
3. **Provides measurable implementation alternatives.** The paper compares fixed and movable ROIs and reports both general OpenCV and optimized CUDA preprocessing, making the cost of region preparation visible rather than assuming it is free.

**Weaknesses:**
1. **Cascading failure risk.** If the fast region focuser fails to detect a candidate region (e.g., a faded or partially occluded sign), the YOLO classifier never receives it, resulting in a complete miss with no recovery mechanism.
2. **Hardware-dependent speed.** Approximately 17 FPS was obtained with three GPUs; a single-GPU movable-ROI configuration was about 7–8 FPS, and a laptop may behave differently.
3. **Dataset and implementation confounds.** The best mAP also used an extended training set, and custom CUDA preprocessing was substantially faster than general OpenCV, so cropping alone does not explain the full result.

---

### Paper 3: Traffic Sign Detection and Recognition Using YOLO Object Detection Algorithm: A Systematic Review

**Summary of Technique:**
Unlike standard experimental research, this paper is a Systematic Literature Review (SLR) that tracks the historical evolution of the YOLO (You Only Look Once) architecture from its early v2 iterations all the way to YOLOv5. With dozens of object detection architectures available today, selecting the correct model for a mobile edge device is a complex balancing act between Mean Average Precision (accuracy) and Frames Per Second (speed).

Using the rigorous PRISMA screening framework, the authors analyzed 115 primary studies to extract comparable performance metrics across different hardware setups. The review highlights specific architectural breakthroughs that have made mobile deployment viable. For instance, it details how the transition to architectural changes in YOLOv5 significantly reduces the computational overhead while actually improving the detection of small objects—like distant traffic signs.

This review provides useful historical context for choosing a lightweight YOLO detector, but it does not prove that one model is universally optimal. Its coverage stops at YOLOv5 and the reviewed experiments used different datasets and hardware. Our current project therefore uses pretrained YOLO26s at a 640-pixel baseline and selects deployment settings from controlled measurements on the target server and laptop-camera recordings rather than from cross-study FPS alone.

**System Block Diagram:**
```text
[Academic Databases]
       │
       ▼
[PRISMA Screening Protocol] ──► (Filter by relevance, year, and YOLO focus)
       │
       ▼
[Architecture Comparison] ──► (Analyze YOLOv2 through YOLOv5)
       │
       ▼
[Metric Extraction] ──► (Compare mAP vs. FPS across hardware)
       │
       ▼
[Candidate Edge Strategies]
```

**Input and Output of Each Block:**
*   **Academic Databases:** Input: Search keywords. Output: Thousands of raw research papers. *(Image Recommendation: Logos of IEEE/ScienceDirect or a search bar screenshot)*
*   **PRISMA Screening Protocol:** Input: Raw papers. Output: 115 filtered, highly relevant primary studies. *(Image Recommendation: Screenshot of the classic PRISMA flowchart from the paper)*
*   **Architecture Comparison:** Input: Selected studies. Output: Structural differences between YOLO generations. *(Image Recommendation: Side-by-side architecture diagram of older YOLO vs YOLOv5)*
*   **Metric Extraction:** Input: Experimental data from studies. Output: Statistical comparison tables of speed vs. accuracy.
*   **Candidate Edge Strategies:** Input: Extracted metrics. Output: A shortlist of lightweight detector options that still require a fair target-device benchmark.

**Text Description:**
The researchers used the PRISMA framework to filter the literature and summarize reported YOLO architectures and metrics. The review supports considering lightweight YOLO variants for constrained hardware, but its numbers are not a head-to-head benchmark because datasets, resolutions, devices and implementations differ. It is therefore evidence for model shortlisting and evaluation, not proof that YOLOv5-nano or any other variant is the only valid choice.

**Strengths:**
1. **Rigorous and reproducible methodology.** The PRISMA systematic review framework ensures transparent selection criteria, making the conclusions auditable and academically defensible — unlike ad-hoc comparisons.
2. **Comprehensive cross-study comparison.** By extracting standardised metrics (mAP and FPS) from 115 primary studies across different hardware platforms and datasets, the review provides a uniquely broad performance landscape that no single experimental paper can offer.
3. **Useful historical synthesis.** The aggregated results show the development of lightweight YOLO approaches and provide reported accuracy/speed ranges that can guide which detector families to benchmark, while the heterogeneous test conditions prevent a universal ranking.

**Weaknesses:**
1. **Does not propose any new algorithm or technique.** As a literature review, this paper only analyses existing work; it contributes no novel architecture, training strategy or dataset of its own.
2. **Coverage stops at YOLOv5.** The review does not include YOLOv6, YOLOv7 or YOLOv8, which were released around the same period or shortly after. This limits the applicability of its conclusions to the very latest architectures.
3. **Heterogeneous experimental conditions across studies.** The compared studies use different datasets, image resolutions, hardware and training configurations, which means the aggregated mAP and FPS averages may not be directly comparable and could mask important performance differences.

---

### Technique Comparison

| Paper | Main Technique | Advantage | Disadvantage | Best Use Context |
| :--- | :--- | :--- | :--- | :--- |
| **Paper 1: Adverse Weather CNN** | Synthetic data augmentation and Spatial Attention Modules. | Motivates training with adverse-condition examples and evaluating attention when ordinary features remain insufficient. | Reported gains are setup-specific; a custom attention module adds implementation and validation work. | Designing and testing robustness for rain, fog, glare and nighttime scenes. |
| **Paper 2: Region Focusing** | Fixed/movable large ROIs, ROI-specific YOLO training and parallel GPU processing. | Can preserve small-sign detail and reduce irrelevant image area under a suitable hardware and implementation setup. | A strict ROI gate can miss signs; measured speed depends on ROI overhead, GPU count and implementation. | A candidate optimization to benchmark on high-definition camera feeds. |
| **Paper 3: YOLO Systematic Review** | Systematic PRISMA literature screening and statistical metric extraction. | Provides historical evidence for shortlisting lightweight YOLO detectors. | Does not propose a new algorithm, stops at YOLOv5 and compares heterogeneous experiments. | Planning a fair model benchmark for edge deployment. |

### Conclusion
Building a traffic sign detection system that works outside a laboratory requires balancing speed and robustness. Paper 1 motivates realistic adverse-condition augmentation, while Paper 2 provides evidence that ROI focusing can be useful when the detector, data, implementation and hardware are designed together. It does not establish that ROI-only inference is always faster or more accurate. Paper 3 supports evaluating lightweight YOLO variants as the CNN detector. Together, the papers motivate a measured hybrid design whose latency, recall and false-positive rate must be verified on the target laptop camera.

---

### Member 4's Proposed Architecture (Future Work Integration)

To demonstrate how these literature review findings translate into our actual project, I have contrasted the original isolated models against my proposed hybrid architecture. As each team member is developing a system block diagram based on their specific research focus, this design represents Member 4's submission. The team will eventually compare all proposed diagrams and choose the best combined design for the final complete report.

#### 1. "Old" (Original) Implementation
Based on the literature reviewed, the "old" or baseline architecture relies on the standard YOLO framework (as seen in Paper 3) without any specific preprocessing or weather-robust features.

**"Old" System Block Diagram:**
```text
[Camera / Video Input (1080p)]
       │
       ▼ (Passes entire massive frame directly to CNN)
[Standard YOLO Network] ──► (Full-frame baseline; speed and robustness must be measured)
       │
       ▼
[Classification Result]
       │
       ▼
[Simple Text Output]
```
**Input and Output of Each Block:**
*   **Camera / Video Input (1080p):**
    *   **Input:** Real-world environment captured by the camera or video file.
    *   **Output:** Massive 1920x1080 RGB image frames.
*   **Standard YOLO Network:**
    *   **Input:** The entire 1920x1080 image frame.
    *   **Output:** Detected bounding boxes and class probabilities (if successful).
*   **Classification Result:**
    *   **Input:** Raw model probabilities.
    *   **Output:** A final text string of the detected sign (e.g., "Stop Sign").
*   **Simple Text Output:**
    *   **Input:** Traffic sign text string.
    *   **Output:** Plain text label displayed to the user.

**Description of the "Old" Implementation:**
This baseline architecture directly feeds each high-definition frame into a standard detector. It is simple and provides a useful accuracy and latency baseline, but it may be slower on limited hardware and may be less robust to rain, blur, backlighting or glare if those conditions are absent from the training data. Its measured performance must be compared with the hybrid alternative on the target laptop.

#### 2. Member 4's Proposed Improved Design (Web-Based Hybrid Architecture)
By synthesizing the techniques from the literature, I propose a safe hybrid server-side web pipeline. It combines **periodic full-frame YOLO26s detection at a 640-pixel baseline** as a recall-preserving safety path, optional **OpenCV Region Focusing** for second-pass crops, realistic **Data Augmentation**, and temporal confirmation. The standard pretrained YOLO26s CNN is retained without a custom backbone or attention module. Its one-to-one head supplies end-to-end, NMS-free detections by default.

The active application is a desktop/laptop web system, not a native mobile application. On an Intel CPU server the validated OpenVINO export is the deployment candidate. YOLO26n is a fallback only if YOLO26s fails the measured latency gate while maintaining the required accuracy; YOLO26m is a challenger only with adequate balanced data and a GPU server.

**Proposed System Block Diagram:**
```text
[1280×720 Browser Webcam / Uploaded Image]
       │
       ▼
[WebSocket / HTTP Upload to Server]
       │
       ▼
[Frame Preparation: preserve aspect ratio; 640 baseline]
       ├──────────────► [Periodic Full-Frame YOLO26s End-to-End Safety Path]
       │
       └──► [Optional OpenCV Colour/Contour Candidates]
                         │
                         ▼
               [ROI + Margin, Letterbox]
                         │
                         ▼
               [YOLO26s End-to-End Second Pass]
       │
       ▼
[Map to Full Frame + Cross-Pass Class / IoU Merge]
       │
       ▼
[Temporal Confirmation] ──► (same class + overlapping box for 2–3 frames)
       │
       ▼
[Confidence Decision]
       ├──► [Confirmed Dashboard / Audio Output]
       └──► [Uncertain / No Announcement]
```

#### 3. Input and Output of Each Block
*   **Browser Webcam / Uploaded Image:**
    *   **Input:** Live environment captured by the user's webcam or an uploaded image/video file.
    *   **Output:** Preferably 1280×720 RGB frames streamed via WebSocket or uploaded via HTTP. *(Image Recommendation: A screenshot of a browser webcam feed showing a street scene)*
*   **WebSocket / HTTP Upload:**
    *   **Input:** Raw frames from the browser.
    *   **Output:** Frames delivered to the Python backend server for processing.
*   **Frame Preparation:**
    *   **Input:** Full frames received by the server.
    *   **Output:** Colour-correctly decoded frames whose geometry is retained; Ultralytics performs aspect-preserving letterboxing rather than stretching the frame to a square.
*   **Periodic Full-Frame YOLO26s Safety Path:**
    *   **Input:** Prepared full frame on a configurable schedule.
    *   **Output:** NMS-free end-to-end full-frame detections that recover signs missed by colour or contour segmentation.
*   **Optional OpenCV Candidate Path:**
    *   **Input:** Prepared full frame.
    *   **Output:** Red, blue or yellow colour/contour candidate boxes. Each box receives contextual margin before its unwarped crop is passed to YOLO26s using aspect-preserving letterboxing at the 640 baseline.
*   **Cross-Pass Merge:**
    *   **Input:** Full-frame and ROI detections mapped into the same frame coordinates.
    *   **Output:** Duplicate-suppressed detections using box overlap and class/confidence evidence. This application-level merge combines separate inference passes; it is not traditional detector NMS.
*   **Temporal Confirmation and Confidence Decision:**
    *   **Input:** Merged detections from consecutive frames.
    *   **Output:** A confirmed sign only when the class and approximately the same box (IoU overlap) persist for 2–3 frames. Low-confidence or inconsistent evidence returns an uncertain/no-announcement state rather than a forced class.
*   **Web Dashboard / Audio Output:**
    *   **Input:** Confirmed sign labels with full-frame bounding boxes.
    *   **Output:** Browser overlays, confidence values, history and a non-repeating alert; uncertain candidates are not announced as signs.

#### 4. Description of the Improved Part
The improvement is a **safe hybrid**, not a strict crop-only cascade. Periodic full-frame inference prevents colour segmentation from becoming a mandatory gate, while optional ROI inference can provide a higher-resolution second view of a small candidate. Mapping both branches to the same coordinates, suppressing duplicates and requiring class-plus-IoU stability reduces flicker and many one-frame false detections.

Realistic camera augmentation (brightness, exposure, mild perspective, blur, sensor noise and compression), laptop-camera source images and background-only hard negatives are used to improve robustness. These choices are expected to help accuracy and noise rejection, but their effect is verified rather than assumed. The team will benchmark pure full-frame YOLO, ROI-only YOLO and the safe hybrid on the same server, videos, 640 input and thresholds.

YOLO26s must reach precision and recall of at least 0.80, mAP@0.5 of at least 0.75, and p95 end-to-end latency of at most 500 ms on the Intel CPU server or 200 ms on the selected GPU server. Its OpenVINO export may lose no more than 0.01 absolute mAP@0.5 against `best.pt`. The safe hybrid is enabled only if it improves p95 latency by at least 10% or small-sign recall by at least 0.02 absolute, while losing no more than 0.01 overall recall and not increasing false positives per minute.
