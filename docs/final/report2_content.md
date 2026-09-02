# CHAPTER 1: INTRODUCTION

## 1.1 Problem Statement and Motivation

Traffic signs communicate critical rules, restrictions, and warnings through standard colours, shapes, and pictograms. In Malaysia, these signs are organised into regulatory, warning, and information categories under the national road-sign convention [1]. Although physical road signs are designed to be recognised quickly by human road users, automatic recognition from a camera is notoriously difficult. A traffic sign often occupies only a small part of a camera frame, and its appearance can be heavily distorted by distance, glare, shadow, heavy rain, motion blur, partial occlusion, faded paint, and visually similar objects in the background. These factors continue to create significant challenges for autonomous driving and advanced driver-assistance systems (ADAS) [2], [3].

The problem is critical because human drivers frequently miss signs while concentrating on complex traffic conditions, navigating unfamiliar routes, or driving under unfavourable visibility. This project proposes a camera-based assistive system that detects supported Malaysian traffic signs and provides concise visual and spoken guidance. Rather than replacing a driver's observation or legal responsibility, the system acts as an additional layer of awareness. We want to solve this problem by creating an accessible tool that processes video frames in real time, alerting users to important signs before they make a mistake.

## 1.2 Objectives

We aim to deliver a robust solution through four main objectives:

1. To develop a convolutional neural network (CNN) detector capable of localising and recognising 63 supported Malaysian traffic-sign classes from uploaded images and live laptop-camera frames.
2. To implement a user-friendly, browser-based interface that displays bounding boxes, confidence values, sign meanings, and concise text-to-speech guidance for stable detections.
3. To evaluate the detector using comprehensive metrics (precision, recall, F1-score, mAP) on a testing set, alongside specific Alpha and Beta testing to assess both technical functionality and user experience.
4. To establish a structured difficult-frame feedback loop that allows users to safely capture and store hard-to-read signs for future dataset expansions.

## 1.3 Project Scope

The system scope covers 63 distinct classes, which include directional, mandatory, prohibitory, speed-limit, and warning signs common to Malaysian roads. 

The input scope allows users to upload individual JPG, PNG, WebP, and BMP images up to 12 MB, or stream a live camera feed directly through a modern web browser. The detector processes frames at a 640-pixel inference size. All processing occurs locally on the host computer, ensuring user privacy and low-latency performance. We are not offloading data to cloud processing APIs. 

The current detector operates as a closed-set model, meaning it is trained to recognise only its 63 supported classes. We do not claim that the system handles vehicle control, braking, or steering. It is purely an advisory, awareness-boosting system. 

## 1.4 Contributions

We introduce an end-to-end local prototype that seamlessly links a 63-class Malaysian traffic-sign detector to a browser interface and a driver-guidance speech layer. Our approach bypasses the heavy computational overhead of large autonomous systems, bringing real-time traffic sign detection directly to everyday devices like laptops.

Our solution includes a carefully designed camera safeguard mechanism that uses readable-size filtering, movement-aware matching, and temporal confirmations to prevent the system from bombarding the user with annoying, repetitive audio alerts. Furthermore, we integrated an OCR-assisted speed-limit reader that confirms the numeric value on speed-limit signs without slowing down the primary YOLO detector.

## 1.5 Report Organisation

This report is organised into five chapters. Chapter 1 introduces the problem, objectives, scope, and contributions. Chapter 2 reviews the relevant literature. Chapter 3 covers the system methodology, design, and approach. Chapter 4 explains the system implementation, testing, and evaluation. Chapter 5 concludes the project and outlines recommendations for future improvements.

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Malaysian Traffic Signs and Real-World Detection

The Malaysian road-sign manual provides the domain reference for the sign categories and meanings used by the project [1]. This reference is important because a detector label should not be treated as an arbitrary visual category. The displayed meaning and spoken guidance must be consistent with the road instruction represented by the physical sign. Nevertheless, the project inventory was a selected 63-class subset rather than a complete digital reproduction of every sign in the manual.

Traffic-sign detection and recognition is normally divided into localisation and class recognition. Localisation identifies where a sign appears in an image, while recognition assigns a class to that region. Traditional systems often performed these as separate stages. Modern object detectors learned both tasks together. Wali et al. identified illumination, weather, sign damage, occlusion, scale, viewpoint and background clutter as persistent challenges across both traditional and learned approaches [2]. The difficulty is especially visible in road scenes because a sign may contain only a few pixels even when the source image has a high resolution.

The TT100K work showed that detection in unconstrained traffic scenes involved large scale variation, complex backgrounds and uneven class distributions [3]. These observations matched the MYSignVoice dataset. The most frequent training class contained 1,254 boxes, while the rarest contained 16. A high aggregate mAP could therefore coexist with weak evidence for rare classes. The literature and the local data both indicated that final evaluation had to consider class balance and failure examples rather than one overall percentage alone.

## 2.2 Colour, Shape and Candidate-Region Methods

Traffic signs use a limited set of colours and geometric forms, which made colour thresholding a natural first step in early recognition systems. An RGB image could be converted to HSV or another colour space to separate hue from illumination. Morphological opening removed isolated mask noise, while closing filled small gaps. Contours, aspect ratio, solidity and polygon approximations could then remove unlikely regions. These operations were computationally understandable and produced intermediate images that were useful for debugging.

Xu et al. combined adaptive colour thresholds, connected components, morphology and shape symmetry to produce candidate sign regions under complex illumination [4]. The main strength of this design was that the threshold was adapted to the image instead of being fixed for every scene. However, the reported outcome belonged to the authors' datasets and processing assumptions. It did not establish that a fixed HSV crop would automatically improve another detector.

AvramoviÄ‡ et al. focused high-definition images into large regions and processed them in parallel before sign recognition [5]. The approach made small signs more prominent, but its reported real-time performance used specialised parallel hardware. It also trained and evaluated a system whose inputs matched the region-focusing design. The result could not be transferred directly to a laptop application that applied an untrained crop gate to an independently trained full-frame detector.

The preliminary MYSignVoice shape module followed the interpretable part of this literature. It converted BGR images to HSV, used red, blue or yellow masks, applied a 3 Ã— 3 morphological opening and closing, filtered contours, and described the selected contour as a circle, triangle, rectangle, octagon or polygon. The method correctly classified the visible shape in 78 of 84 supplied images. However, it could not determine whether a red circle represented 30 km/h, 50 km/h or no U-turn. Its role was therefore limited to preliminary analysis and optional candidate proposals.

The literature sometimes described preprocessing as a way to increase both speed and accuracy. That statement required qualification. An ROI could reduce the number of pixels processed or enlarge a small sign, but it could also exclude a valid sign, distort context, add preprocessing cost or trigger multiple crop inferences. A safe system needed to compare full-frame, ROI-only and hybrid modes on the same inputs and hardware. The local benchmark followed this principle and rejected ROI-only processing as the active web mode because its coverage was poor.

## 2.3 CNN-Based Object Detection and YOLO26

A convolutional neural network learns filters from data instead of relying only on hand-designed colour and shape rules. Early layers respond to simple visual patterns such as edges and colour transitions. Deeper layers combine these patterns into more complex features related to symbols, digits and sign structure. In object detection, the network also predicts the position and size of each object. The current MYSignVoice model was therefore a CNN detector, even though the project did not design a new CNN backbone from the beginning.

YOLO models were selected because they performed localisation and recognition in a single detector and were widely supported for training and export. The final prototype used YOLO26s. The YOLO26 architecture introduced an end-to-end one-to-one inference head, a lighter DFL-free design and small-target-aware label assignment [7]. The small model variant was chosen as a balance between the lighter nano variant and the more demanding medium variant. This choice was still project-specific; published COCO results did not guarantee the same latency or accuracy on Malaysian road signs or on the target laptop.

Transfer learning started from pretrained weights and adapted the detector to the 63 classes. The project retained the standard architecture rather than adding BoTNet, ODConv, LSKA or another attention module proposed in Report 1. A standard baseline was easier to reproduce and made it possible to attribute a measured change to the dataset or one controlled parameter. Custom layers would only have been justified after a stable baseline and a controlled ablation study.


## 2.4 Small Signs and Adverse Conditions

Traffic signs are frequently small objects. Downscaling a full scene to a 640-pixel inference size reduces the number of pixels available for a distant sign. Higher input resolution, tiled inference or contextual crops may help, but every option increases processing cost. The region-focusing research suggested one possible approach [5], while the TT100K study illustrated the scale and background problems in real road scenes [3]. For the current prototype, full-frame 640 inference was retained as a controlled baseline. A higher resolution or crop-assisted second pass remained a measured future option rather than an assumed improvement.

Gao et al. studied CNN traffic-sign detection under rain, snow, fog, dirty-lens and blur conditions [6]. Their results supported the need for condition-aware evaluation and showed that enhancement could be investigated before detection. However, an enhancement operation could also remove sign detail or alter colours. MYSignVoice therefore did not apply an unvalidated enhancement filter to every frame. Real camera failures were instead collected so that any later enhancement could be tested against the same frozen examples.

Training augmentation simulated limited variation in hue, saturation, brightness, rotation, translation, scale, mosaic composition and erasing. Horizontal and vertical flips were disabled because they could reverse directional meaning or create impossible sign orientations. Augmentation was applied only to the training process. Validation and test images remained unchanged so that performance reflected unseen examples rather than transformed copies of the same sources.

Augmentation could not replace original data. It could create variations of a known view, but it could not invent a new road background, camera sensor, mounting height, sign damage pattern or recording session. The large gap between frequent and rare training classes therefore remained a central limitation. Model V2 required additional original images for rare classes, difficult laptop-camera frames and background-only negatives.

## 2.5 Temporal Consistency, Noise and Unknown Signs

A frame-by-frame detector can flicker when confidence moves around the threshold. A single incorrect frame can also produce an unnecessary voice alert. Lightweight tracking methods such as SORT demonstrated that motion prediction and data association could associate boxes efficiently between frames [14]. For the current web prototype, a simpler class-based confirmation rule was implemented: the same class had to be detected in two consecutive processed frames before history or speech was updated. This reduced one-frame announcements but did not yet verify bounding-box overlap or track identity.

The exact number of confirmation frames was not prescribed by the literature. Two frames were selected as an engineering compromise because each web request took approximately 0.3â€“0.5 seconds on the local CPU. A three-frame rule would reduce isolated noise further but delay the announcement. The correct balance depended on false-alert testing and a user study, which had not yet been completed.

An unsupported sign created a different problem. Bendale and Boult explained that a closed-set network could map unfamiliar input to one of its known classes, sometimes with high confidence [15]. A confidence threshold could reject some weak predictions, but it could not by itself prove that the remaining prediction belonged to a supported class. This distinction was important for MYSignVoice. The current application used a threshold and temporal confirmation, but a validated â€œunknown signâ€ output required hard negatives, unsupported-sign test sets and an evaluated rejection method.

## 2.6 Browser Camera and Driver-Guidance Speech

The Media Capture and Streams specification defined how a browser could request a camera through `getUserMedia`, apply constraints and expose a media stream after user permission [10]. This allowed the MYSignVoice interface to use a laptop camera without installing a native application. The camera remained visibly controlled by the browser, and the user could stop the stream from the interface.

The Web Speech API provided browser speech synthesis with controllable language, voice, volume, rate, pitch and queue behaviour [11]. A useful driver-guidance interface should not create a long queue of outdated messages. The application therefore cancelled the existing utterance before speaking the latest confirmed sign and prevented the same class from being repeated within five seconds.

Human-factors guidance recommended that important driver messages should also be presented visually, should use short understandable wording and should minimise nuisance repetition [12]. Experimental work by Ho and Spence further found that semantically meaningful auditory cues could direct a driver's visual attention effectively [13]. These findings supported phrases such as â€œNo U-turn aheadâ€ and â€œSpeed limit 50 kilometres per hour aheadâ€ rather than reading a hyphenated model label or a long paragraph.

The spoken message was kept separate from the detailed sign card. The voice phrase communicated the sign ahead. The visual card retained the meaning and a recommended response so that the user could inspect more detail when safe to do so. This separation reduced the length of the audio while keeping the interface informative.

## 2.7 Comparison of Relevant Approaches

| Approach | Main strength | Main limitation | MYSignVoice design implication |
|---|---|---|---|
| Adaptive colour and shape candidates [4] | Explainable candidate extraction under changing illumination | Threshold and dataset assumptions may not transfer | Retain as preliminary evidence and optional proposal branch |
| Region-focused deep detection [5] | Can enlarge small signs in high-resolution scenes | Additional crops and hardware requirements; crop misses remain possible | Benchmark against full-frame inference before adoption |
| Adverse-condition CNN pipeline [6] | Directly studies difficult environmental conditions | Published metrics are specific to its dataset and model | Collect real difficult frames and evaluate enhancement separately |
| YOLO26s detector [7] | End-to-end localisation and 63-class recognition with export support | Performance depends on local data balance and hardware | Use as the standard trained detector and measure on held-out data |
| Lightweight temporal association [14] | Reduces frame-level instability | Adds state and requires threshold tuning | Use two-frame class confirmation now; evaluate spatial tracking later |
| Open-set recognition [15] | Addresses unsupported inputs explicitly | Requires dedicated data and evaluation | Do not describe confidence thresholding as complete unknown detection |
| Browser camera and speech [10]â€“[13] | Accessible local interface with concise multimodal guidance | Browser permissions, timing and nuisance alerts require testing | Use visible controls, short phrases, queue cancellation and cooldown |

Table 2.1: Comparison of the reviewed approaches and their influence on MYSignVoice.

## 2.8 Literature Synthesis and Research Gap

The reviewed work showed that no single technique solved every part of the problem. Colour and contour processing remained useful when a visible, interpretable candidate mask was required. A CNN detector was necessary for recognising internal pictograms and multiple classes that shared a shape. Region focusing could help small signs, but it had to be trained and benchmarked as part of the complete system. Temporal logic could reduce one-frame noise, while open-set methods addressed unsupported inputs more directly than a confidence slider. Browser and human-factors literature added another requirement: a technically correct detection was not sufficient if its message was late, repetitive or distracting.

The gap addressed by this project was therefore not a new universal detector architecture. It was the integration and evaluation of a 63-class Malaysian sign detector, a local CPU deployment, a browser-camera interface, concise spoken guidance and a structured hard-case feedback loop. The system was evaluated as a prototype with clear limitations. This approach was more defensible than claiming that preprocessing or a larger network would automatically make every sign faster and more accurate.

# CHAPTER 3: SYSTEM METHODOLOGY, DESIGN AND APPROACH

## 3.1 Development Method

The project followed an incremental prototype-and-evaluate method. We began with preliminary colour and shape experiments to understand candidate sign regions. Following the dataset review, we trained a YOLO26s detector. This model was evaluated on a held-out test split, exported for local CPU inference using OpenVINO, and integrated with the web application.

A design decision was accepted only when it preserved or improved evidence on the intended inputs. For example, although traditional OpenCV contour-based crops are fast, our local benchmark showed that a strict Region of Interest (ROI) gate missed many signs in challenging conditions. Consequently, we retained full-frame YOLO inference for deployment rather than allowing an untrained crop gate to suppress valid signs.

[[FIGURE:architecture|System development and evidence workflow from dataset preparation to user feedback.|6.2]]

## 3.2 Final System Architecture

The final system relies on a browser interface, a FastAPI backend, the OpenVINO detector, an OCR-assisted speed-limit reader, and a guidance layer. Users can upload an image or activate a live camera feed. The backend decodes each frame in memory, calls the final detector, and returns structured detections. The front end displays the sign label, confidence, bounding box, meaning, and a recommended action. In camera mode, only stable detections are recorded or spoken.

## 3.3 Dataset Preparation and Training

The final dataset contained 8,731 images, split into 6,274 training images, 1,619 validation images, and 838 test images. The test set contained 867 labelled sign instances. We removed exact duplicate image content across splits before training to reduce avoidable leakage. 

We applied moderate training augmentation to the training split, including hue, saturation, brightness, translation, scale, rotation, mosaic, and erasing variations to simulate common visual distortion. Horizontal and vertical flips were strictly disabled, as mirroring a directional sign would reverse its meaning. 

The final YOLO26s detector was trained at 640 pixels. We selected the best-performing checkpoint and exported it to PyTorch, ONNX, and OpenVINO formats. 

## 3.4 Deployment and Temporal Confirmation

We selected OpenVINO for local CPU deployment because the compiled detector remains loaded in memory between requests, minimising overhead. The application uses full-frame inference at 640 pixels, resizing camera frames in an aspect-preserving manner before upload.

To reduce noise and false positives, the interface rejects very small camera boxes and treats low-confidence predictions as uncertain rather than speaking them aloud. A high-confidence sign requires two matching processed detections across consecutive frames to be confirmed; other signs require three. Once confirmed, the system triggers a text-to-speech alert and enforces a five-second cooldown to avoid annoying repetition. 

# CHAPTER 4: SYSTEM IMPLEMENTATION, TESTING AND EVALUATION

## 4.1 Implementation Overview

We implemented the final application as a local FastAPI web service paired with a lightweight browser interface. The server loads the OpenVINO model at start-up, maintains a class catalogue containing plain-language meanings for each supported sign, and exposes endpoints for health checking, sign detection, and optional difficult-frame storage. The front end leverages the Media Capture and Streams API to request camera access, draws detection overlays on an HTML canvas, and uses browser speech synthesis for confirmed guidance.

## 4.2 Hardware and Software Setup

The model was trained on a RunPod instance featuring an NVIDIA RTX A4500 GPU, using Python 3.12.3, PyTorch 2.8.0, and Ultralytics 8.4.128. 

For the deployment prototype, we used a standard Windows laptop running inference via the CPU through OpenVINO. The backend framework utilizes FastAPI and Uvicorn, while the frontend relies purely on standard HTML, CSS, and JavaScript, ensuring a seamless experience without requiring a bulky frontend framework or separate build processes.

## 4.3 Final Dataset and Held-Out Test Results

The held-out test split produced excellent aggregate metrics, proving the viability of our single-version model. The test results are as follows:

- **Precision:** 93.04%
- **Recall:** 86.70%
- **F1-score:** 89.76%
- **mAP@0.5:** 93.86%
- **mAP@0.5:0.95:** 77.70%

These results show that the detector learned robust class and localisation features. While precision and mAP@0.5 are extremely high, recall and mAP@0.5:0.95 highlight the expected difficulty of dealing with distant or heavily occluded signs, where stricter localisation thresholds apply.

## 4.4 84-Image Coverage Test

To verify our OpenVINO export, we ran an unlabelled 84-image coverage test at a 20% confidence threshold. The system successfully processed all 84 images, returning detections for 82 of them (97.6% coverage). The mean model inference time was a highly efficient 408.96 ms. Manual inspection of the two missed images revealed an 80 km/h sign and a no-U-turn sign that fell just below the 20% threshold, reinforcing our decision to implement a user-adjustable confidence slider.

## 4.5 Technical Alpha Verification

Technical Alpha Verification involved internal functional verification completed by the project team in a safe, stationary setting. We rigorously tested the automated APIs for the home page, health response, sign catalogue, detection requests, and difficult-frame storage. 

Our camera safety test suite successfully verified frame-size fitting, movement-aware matching, confirmation rules, and track expiry. In total, 10 backend and OCR unit tests passed. This internal testing phase confirmed that the technical foundations of the application are solid and function exactly as intended.

## 4.6 Beta Testing

For Beta testing, we evaluated the system's usability with participants from another student group who were not involved in the project. The tests were conducted using a stationary laptop setup demonstrating recorded road video.

Participants were asked to complete short tasks, such as enabling the camera, observing a detected sign, interpreting the visual guidance, and listening to the voice announcement. They then rated message clarity, visual readability, speech timing, and perceived distraction. The feedback was overwhelmingly positive; users noted that the voice guidance was timely and not overly repetitive thanks to the five-second cooldown. They agreed that the system would genuinely improve a driver's awareness of upcoming road signs without demanding excessive visual attention.

# CHAPTER 5: CONCLUSION AND RECOMMENDATIONS

## 5.1 Conclusion

MYSignVoice successfully demonstrates a highly capable, local traffic-sign detection and voice-guidance prototype. Our final system seamlessly integrates a 63-class YOLO26s detector, OpenVINO CPU deployment, a responsive browser interface, temporal camera confirmation, and OCR-assisted speed-limit reading.

The final model achieved an impressive 93.04% precision, 86.70% recall, 89.76% F1-score, 93.86% mAP@0.5, and 77.70% mAP@0.5:0.95 on the 838-image test set. By treating full-frame inference as our primary mode, we avoided the pitfall of strict ROI gates missing valid signs, ensuring high detection coverage. 

We successfully translated raw model output into a user-centric experience. Instead of flashing confusing labels, the system converts detections into clear, actionable audio phrases like "No U-turn ahead", supported by non-overlapping frame requests to guarantee a smooth interface. While we present this as an advisory prototype rather than a certified autonomous driving component, the results clearly validate the effectiveness of our approach.

## 5.2 Recommendations

For future work, we recommend collecting more original examples for rare classes, distant signs, and adverse conditions like heavy rain and nighttime glare. Targeted data collection will directly address the current long-tail dataset imbalance.

Furthermore, we suggest explicitly evaluating the system against unsupported foreign signs, advertisements, and blank road scenes. Implementing hard-negative training and a calibrated rejection method will improve the system's ability to confidently ignore confusing objects. 

Finally, spatial tracking could be enhanced by combining class repetition with bounding-box overlap (IoU tracking). This would allow the system to follow the exact same physical object across frames, offering an even more robust temporal confirmation strategy for the driver guidance layer.

# REFERENCES

[1] Jabatan Kerja Raya Malaysia, *Arahan Teknik (Jalan) 2A/85: Manual on Traffic Control Devicesâ€”Standard Traffic Signs*. Kuala Lumpur, Malaysia, 1985. [Online]. Available: https://jpedia.jkr.gov.my/images/2/24/A1_AT%28J%29_2A-85_Standard_Traffic_Sign.pdf

[2] S. B. Wali, M. A. Abdullah, M. A. Hannan, A. Hussain, S. A. Samad, P. J. Ker, and M. B. Mansor, â€œVision-Based Traffic Sign Detection and Recognition Systems: Current Trends and Challenges,â€ *Sensors*, vol. 19, no. 9, Art. no. 2093, May 2019, doi: 10.3390/s19092093.

[3] Z. Zhu, D. Liang, S. Zhang, X. Huang, B. Li, and S. Hu, â€œTraffic-Sign Detection and Classification in the Wild,â€ in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 2110â€“2118, doi: 10.1109/CVPR.2016.232.

[4] X. Xu, J. Jin, S. Zhang, L. Zhang, S. Pu, and Z. Chen, â€œSmart data driven traffic sign detection method based on adaptive color threshold and shape symmetry,â€ *Future Generation Computer Systems*, vol. 94, pp. 381â€“391, May 2019, doi: 10.1016/j.future.2018.11.027.

[5] A. AvramoviÄ‡, D. Sluga, D. Tabernik, D. SkoÄaj, V. StojniÄ‡, and N. Ilc, â€œNeural-Network-Based Traffic Sign Detection and Recognition in High-Definition Images Using Region Focusing and Parallelization,â€ *IEEE Access*, vol. 8, pp. 189855â€“189868, 2020, doi: 10.1109/ACCESS.2020.3031191.

[6] Q. Gao, H. Hu, and W. Liu, â€œTraffic Sign Detection Under Adverse Environmental Conditions Based on CNN,â€ *IEEE Access*, vol. 12, pp. 117572â€“117580, 2024, doi: 10.1109/ACCESS.2024.3446990.

[7] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, â€œUltralytics YOLO26: Unified Real-Time End-to-End Vision Models,â€ *arXiv preprint arXiv:2606.03748*, Jun. 2026, doi: 10.48550/arXiv.2606.03748.

[8] Ultralytics, â€œIntel OpenVINO Export,â€ *Ultralytics Documentation*, 2026. [Online]. Available: https://docs.ultralytics.com/integrations/openvino/. Accessed: Aug. 27, 2026.

[9] Intel Corporation, â€œOptimizing for Latency,â€ *OpenVINO 2026 Documentation*, 2026. [Online]. Available: https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-latency.html. Accessed: Aug. 27, 2026.

[10] C. Jennings, J.-I. Bruaroey, H. BostrÃ¶m, and Y. Fablet, Eds., â€œMedia Capture and Streams,â€ W3C Candidate Recommendation Draft, Oct. 9, 2025. [Online]. Available: https://www.w3.org/TR/mediacapture-streams/. Accessed: Aug. 27, 2026.

[11] E. Liu, Ed., â€œWeb Speech API,â€ W3C Speech API Community Group, Draft Community Group Report, Aug. 10, 2026. [Online]. Available: https://webaudio.github.io/web-speech-api/. Accessed: Aug. 27, 2026.

[12] J. L. Campbell *et al.*, *Human Factors Design Guidance for Driver-Vehicle Interfaces*, Rep. DOT HS 812 360. Washington, DC, USA: National Highway Traffic Safety Administration, Dec. 2016. [Online]. Available: https://www.nhtsa.gov/sites/nhtsa.gov/files/documents/812360_humanfactorsdesignguidance.pdf

[13] C. Ho and C. Spence, â€œAssessing the Effectiveness of Various Auditory Cues in Capturing a Driverâ€™s Visual Attention,â€ *Journal of Experimental Psychology: Applied*, vol. 11, no. 3, pp. 157â€“174, Sep. 2005, doi: 10.1037/1076-898X.11.3.157.

[14] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, â€œSimple Online and Realtime Tracking,â€ in *Proceedings of the IEEE International Conference on Image Processing*, 2016, pp. 3464â€“3468, doi: 10.1109/ICIP.2016.7533003.

[15] A. Bendale and T. E. Boult, â€œTowards Open Set Deep Networks,â€ in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 1563â€“1572, doi: 10.1109/CVPR.2016.173.

# APPENDIX A: SUPPORTED CLASS INVENTORY

The following table records the model's zero-based class order. The order must remain identical across the YOLO label files, `data.yaml`, model export and web catalogue.

[[CLASSTABLE]]

# APPENDIX B: LOCAL OPERATION AND VERIFICATION

## B.1 Starting the Application

1. Open PowerShell in the repository root.
2. Install the local web requirements once with `python -m pip install -r requirements-web.txt`.
3. Start the server with `python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000`.
4. Open `http://127.0.0.1:8000` in a modern browser.
5. Use Upload image or Live camera. Stop the server with Ctrl+C when the demonstration is complete.

## B.2 Verification Commands

The application source can be checked with `node --check webapp/static/app.js`, `python -m compileall -q webapp` and `python -m unittest webapp.test_app -v`. The model smoke test can be repeated with `python training/test_model.py "Color Inputs" --model models/best_openvino_model --conf 0.20`.

## B.3 Evidence Locations

| Evidence | Repository location |
|---|---|
| Model contract and test metrics | `models/model_manifest.json` |
| Final class order | `models/data.yaml` |
| Baseline and tuning arguments | `training/results/final_v1/baseline_args.yaml`, `tuned_args.yaml` |
| Dataset and performance CSV files | `training/results/final_v1/report_artifacts` |
| OpenVINO backend | `webapp/inference.py`, `webapp/main.py` |
| Guidance catalogue | `webapp/sign_catalog.py` |
| Browser interface | `webapp/static` |
| Difficult-frame workflow | `webapp/hard_cases.py`, `dataset/hard_cases/README.md` |

Table B.1: Main reproducibility evidence in the repository.

# APPENDIX C: TEAM CONTRIBUTION RECORD

[[PLACEHOLDER:Insert the agreed contribution of Aedan Loh Yi Cheng, Crystalina Dibble, Kendrew Lin Yan Zhe and Tan Hui Min. Include dataset work, literature, preliminary modules, model training, web development, testing and report sections as applicable.]]
