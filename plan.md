# UCCC2513 Mini Project — Full 14-Week Plan

# 🚦 Intelligent Malaysian Traffic Sign Recognition Using Real-Time Web-Based Detection

---

## 📋 Project Overview

**Course**: UCCC2513 Mini Project
**Team Size**: 4 members
**Duration**: 14 weeks
**Skill Profile**: C++ strong, web development & deep learning are new (plan is written for rookies in web dev/ML)
**Assessments**:
- Assignment (Proposal Report + Presentation): **30 marks** — due ~Week 7
- Class Participation + Activities: **10 marks** — ongoing
- Project Work (Final Report + Presentation): **50 marks** — due ~Week 11–13

**Testing Environment**:
- **Primary**: Chrome, Firefox or Edge on the target desktop/laptop
- **Demo/Review on PC**: Open the server-side web app, use the laptop webcam or upload test images/videos
- **Out of scope**: native Android/iOS/mobile application deployment

**GPU for Training**:
- **No local training GPU needed** — use **Google Colab** for the YOLO26s baseline. Record actual training duration rather than assuming a fixed time.

---

## 🏷️ Suggested Project Titles

> The title must clearly state: **traffic sign detection** + **autonomous vehicles / assisted mobility** (as per assignment brief).

| # | Title | Why It Works |
|---|-------|-------------|
| 1 | **Intelligent Malaysian Traffic Sign Recognition Using Real-Time Web-Based Detection** | Clear, direct — web-based real-time detection, strong technical focus |
| 2 | **Real-Time Malaysian Traffic Sign Detection and Classification System Using YOLO26 on a Web Platform** | Emphasizes real-time + YOLO + web platform |
| 3 | **MYSignWeb: A Web-Based Malaysian Road Sign Detection System for Driving Assistance** | Branded name (MYSignWeb), clearly describes what the system does |

> **Recommendation**: Title **1** or **3**. Title 1 is more formal/academic. Title 3 has a catchy brand name.

---

## 💡 The System — Real-Time Web-Based Traffic Sign Detector

### Concept

A server-side web application that uses a laptop-camera feed or uploaded video/image to detect and classify Malaysian road signs. A Python backend (Flask/FastAPI) runs YOLO26s at a 640-pixel baseline, and a desktop/laptop browser displays bounding boxes, class labels, confidence scores and detection history. No native mobile app is part of the active system.

### Why This Idea Works

The assignment frames the proposal as competing for RM 1,000,000 funding. This idea works because:
- **Practical impact** — improves road safety awareness for all Malaysian drivers
- **Clear innovation** — real-time web-based Malaysian road sign detection system
- **Naturally justifies every technical choice**:
  - **Why camera?** — Real-time detection from live video feeds
  - **Why web app?** — Centralized server inference, no client installation, easy deployment and scaling
  - **Why real-time?** — Instant feedback for driving assistance scenarios
  - **Why Malaysian signs?** — Localized detection tool with genuine practical need
  - **Why C++ + YOLO?** — Performance-critical real-time inference, C++ for preliminary OpenCV work

### Complete Feature List

| Feature | Description | Technical Component |
|---------|-------------|-------------------|
| **Real-time webcam detection** | Live detection from a laptop browser webcam | `getUserMedia` + throttled HTTP + YOLO26s |
| **Image/video upload** | Upload a photo or video for batch detection | HTTP upload + YOLO26s |
| **Bounding box overlay** | Draw detected sign bounding boxes on the video/image | HTML Canvas / JavaScript |
| **Class labels + confidence** | Display sign name and confidence percentage | JSON response from backend |
| **Detection history log** | Scrollable list of recent detections with timestamps | Frontend state management |
| **Difficult-frame collection** | Save user-confirmed misses, wrong classes and false positives for later annotation | Local image folder + JSONL metadata |
| **Confidence threshold slider** | Adjustable sensitivity (default 0.20, range 0.10–0.70) | Settings panel |
| **Sign meaning database** | JSON database mapping 63 sign IDs → name → description | Server-side JSON file |
| **Export results** | Download detection results as CSV or annotated images | Backend export endpoint |
| **Desktop web design** | Works in Chrome, Firefox and Edge on the target desktop/laptop | CSS web layout; no native mobile app |
| **Multi-sign detection** | Detect multiple signs in a single frame | YOLO multi-object output |

### Web App Screen Flow

```
[Landing Page] → [Detection Dashboard] → [Settings Panel]
                        ↓                       ↓
                 [Webcam Live View]      [Confidence Slider]
                 [Upload Image/Video]    [Model Info]
                 [Bounding Box Overlay]
                        ↓
                 [Detection History]
                 [Export Results]
```

### Malaysian Road Signs to Support (63 Classes)

The canonical zero-based IDs and 63 hyphenated names are in `dataset/data.yaml`.
The reviewed 47-class seed list occupies IDs 0–46; the approved Malaysia-road-sign
expansion occupies IDs 47–62. Do not infer or add classes from generic traffic-sign
lists. Corrected class ID 32 is `pass-obstacle-on-either-side`.

| Inventory group | Examples from the locked list | Exact class count |
|---|---|---:|
| **Blue signs** | straight-only, left/right directions, pass-right, roundabout, cars-only, use-horn, bicycle-path | 11 |
| **Red signs** | seven supported speed limits, prohibitions, stop-sign, no-entry, give-way, stop-for-inspection | 21 |
| **Yellow signs** | traffic-light/general warnings, crossings, turns, descent, construction, slippery road, railway warnings | 15 |
| **Approved expansion** | bumps, bus-stop, camera-operation-zone, cow-nearby-warning, limits/parking/towing, chevrons, crossroad/road-narrow/diverge warnings, reverse-turn-warning | 16 |
| **Total** | See `dataset/data.yaml` for all exact names and IDs | **63** |

---

## 🛠️ Tech Stack

### Core Development

| Component | Technology | Why | Beginner Tip |
|-----------|-----------|-----|-------------|
| **Core Logic** | **C++17** | Required by course. Used for preliminary image processing | You already know this well |
| **Web Backend** | **Python + FastAPI** | Loads OpenVINO once and serves image/frame inference | `/api/health` confirms readiness |
| **Web Frontend** | **HTML + CSS + JavaScript** | Camera access, display results, interactive UI | Standard web technologies |
| **Real-time Communication** | **Throttled HTTP POST** | Sends one compressed webcam frame at a time without a request queue | Simple and adequate for the measured CPU latency |
| **ML Inference** | **Ultralytics YOLO26s** | Default end-to-end, NMS-free server inference at 640 | Validate `best.pt` before export |
| **Intel CPU deployment** | **OpenVINO** | Optimize and serve the validated YOLO26 export on Intel hardware | Compare export accuracy and latency |
| **Image Processing** | **OpenCV (Python)** | Resize, convert, draw bounding boxes server-side | Same OpenCV you used in C++, but in Python |

### Web Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Web Browser (Frontend)                  │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Webcam      │  │  Detection   │  │  History   │ │
│  │  (WebRTC)    │  │  Overlay     │  │  Log       │ │
│  └──────┬──────┘  └──────▲───────┘  └─────▲──────┘ │
│         │                │                │         │
│         ▼                │                │         │
│  ┌──────────────────────────────────────────────┐   │
│  │          Throttled HTTP Connection            │   │
│  └──────────────────────┬───────────────────────┘   │
└─────────────────────────┼───────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────┐
│              Python Backend (Flask/FastAPI)           │
│                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  OpenCV   │  │  YOLO26s     │  │   Sign       │  │
│  │ (resize,  │  │  (Ultralytics│  │  Database    │  │
│  │  convert) │  │   inference) │  │  (JSON)      │  │
│  └──────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 🤖 YOLO26 Server Model Decision

### Controlled model ladder

| Candidate | Role | When it is accepted |
|---|---|---|
| **YOLO26s** | Default 640 baseline | Passes held-out quality and target-server latency gates |
| **YOLO26n** | Intel CPU latency fallback | YOLO26s misses latency, while YOLO26n still passes every minimum quality gate |
| **YOLO26m** | Accuracy challenger | Adequate balanced original data and a GPU server exist, and measured quality improves without breaking latency/false-positive gates |

YOLO26's default one-to-one head produces end-to-end detections without external
detector NMS. Keep that default for training validation, prediction and export.
The safe-hybrid pipeline may still use class-aware IoU to combine full-frame and
ROI results; that is application-level cross-pass deduplication, not detector NMS.

Start all controlled model comparisons at `imgsz=640`. On an Intel CPU server,
export the selected model to OpenVINO and verify accuracy parity and end-to-end
latency. The active workflow is server-side web deployment; the Android guide and
ncnn converter are archival and not part of the build.

Official references: [YOLO26](https://docs.ultralytics.com/models/yolo26/),
[end-to-end detection](https://docs.ultralytics.com/guides/end2end-detection/), and
[OpenVINO](https://docs.ultralytics.com/integrations/openvino/).

---

## ☁️ Google Colab Training Guide

Since the team has no local GPU, use Google Colab (free).

### Step-by-Step

1. Go to **https://colab.research.google.com**
2. Create a new notebook
3. **Change runtime**: `Runtime` → `Change runtime type` → select `T4 GPU`
4. Upload your dataset (or mount Google Drive)
5. Run:

```python
# Cell 1: install the tested dependency range.
!pip install -q "ultralytics>=8.4.90,<9" roboflow pyyaml openvino onnx onnxruntime

# Cell 2: mount Drive so the complete run survives a Colab reset.
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: after cloning/uploading this repository, run the canonical script.
# Store ROBOFLOW_API_KEY in Colab Secrets/environment first.
%cd /content/miniproject/training
!python train_colab.py \
  --workspace "your-workspace" \
  --project "mysignvoice-49-signs" \
  --version 1 \
  --model yolo26s.pt \
  --imgsz 640 \
  --epochs 150 \
  --batch 16
```

### Training Planning Estimate (Google Colab T4 GPU)

| Dataset Size | Epochs | Approx Time |
|-------------|--------|-------------|
| 200 images | 100 | ~15 min |
| 500 images | 100 | ~30 min |
| 1000 images | 100 | ~60 min |
| 2000 images | 150 | ~2 hours |

These are planning estimates only. Record the actual GPU type, package version,
batch size, epoch count and wall-clock time for the final run.

### Important Colab Tips

- Free tier has a **~4 hour** session limit — save checkpoints to Google Drive
- Training auto-saves checkpoints every epoch, so if disconnected you can resume
- The canonical script uses `patience=30` and saves the best checkpoint even if training stops early

### Measured acceptance gates

Use a frozen test split and the same saved laptop-camera videos for all candidates:

- accept YOLO26s when precision and recall are each at least 0.80, mAP@0.5 is at
  least 0.75, and p95 end-to-end latency is at most 500 ms on the Intel CPU server
  or 200 ms on the selected GPU server;
- accept the OpenVINO export only if it loses no more than 0.01 absolute mAP@0.5
  versus its `best.pt` source;
- use YOLO26n only when YOLO26s misses latency and YOLO26n still passes the
  minimum quality gates;
- test YOLO26m only with adequate balanced data and a GPU, accepting it only for
  at least 0.02 absolute improvement in mAP@0.5:0.95 or small-sign recall without
  breaking latency or false-positive limits; and
- enable the safe hybrid only if it improves p95 latency by at least 10% or
  small-sign recall by at least 0.02 absolute, while losing no more than 0.01
  overall recall and not increasing false positives per minute.

---

## 📅 14-Week Schedule

> Aligned with the **Teaching Plan** milestones (Week 2: title submission, Week 5: lit review draft, Week 7: proposal submission + presentation, Week 11–13: final presentation).

### Phase 1: Research & Proposal (Weeks 1–7)

| Week | Focus | Deliverables | Who |
|------|-------|-------------|-----|
| **1** | **Project Kickoff** | • Form team, assign roles • Read all course documents • Read this entire plan.md • Set up GitHub repo + WhatsApp/Discord group • Each member installs Python + VS Code | All |
| **2** | **Title & Literature Search** | • Submit project title + team list to lecturer • Each member searches 5–8 papers on their assigned keywords • Start shared Google Doc for references | All |
| **3** | **Deep Literature Review** | • Each member writes 3-page critique for their area with proper citations • Draft Chapter 2 (Literature Review) • Group meeting to discuss findings and agree on system design | All |
| **4** | **Setup Dev Environment** | • Install & configure: Python, Flask, OpenCV, Ultralytics, OpenVINO, VS Code • Run a "Hello World" Flask web server (Member 1 & 2) • Run a YOLO26s inference test on a sample image in Python (Member 3 & 4) • Start photographing/downloading Malaysian road sign images | Member 1 & 2 (Web), Member 3 & 4 (ML/Python) |
| **5** | **Draft Proposal Chapters 1–3** | • Chapter 1 (Introduction) — *Member 1* • Chapter 2 (Lit Review) — combine all 4 reviews — *All* • Chapter 3 (Proposed Method) — system block diagram + flowchart — *Member 2* • Submit draft of Chapter 2 to supervisor | All |
| **6** | **Preliminary Work (Ch 4) + Finalize Proposal** | • Each member implements their preliminary module in C++ (see Ch 4 modules below) • Chapter 4 screenshots and results • Chapter 5 (Conclusion) — *Member 3* • Proofread, Turnitin check, format per FYP1 template | All |
| **7** | **🎯 PROPOSAL SUBMISSION + PRESENTATION** | • Submit proposal report (Ch 1–5) • Record/present presentation video with demo • Each member shows their segmentation results on test images | All |

### Phase 2: Development (Weeks 8–10)

| Week | Focus | Deliverables | Who |
|------|-------|-------------|-----|
| **8** | **Dataset + Model Training** | • Complete and audit the 63-class dataset • Apply training-only augmentation • Train YOLO26s at 640 on Google Colab • Evaluate held-out metrics • Export and validate OpenVINO for Intel CPU | Member 3 (dataset) + Member 4 (training), Member 1 & 2 (help annotate + start web scaffold) |
| **9** | **Web App Core** | • FastAPI backend with OpenVINO inference endpoint (Member 2) • Throttled HTTP webcam-frame submission (Member 2) • Frontend: webcam view + bounding box canvas overlay (Member 1) • Sign database JSON (Member 3) • Model testing pipeline (Member 4) • **Integration day**: combine frontend → backend → YOLO → display | Member 1 (frontend), Member 2 (backend), Member 3 (database), Member 4 (model) |
| **10** | **Features + Testing** | • Image/video upload working • Detection history log • Confidence threshold slider • Test on all 84 provided images → log recognition rate • Test on uploaded videos from real roads • Bug fixes, performance tuning (<2 sec requirement) | All |

### Phase 3: Final Report & Presentation (Weeks 11–14)

| Week | Focus | Deliverables | Who |
|------|-------|-------------|-----|
| **11** | **Final Report Writing** | • Follow FYP2 report template • Expand Ch 1–3 from proposal • Ch 4 (Implementation): each member documents their module • Ch 5 (Results): recognition rates, confusion matrix, error analysis | All |
| **12** | **Report Completion** | • Ch 6 (Conclusion + Future Work) — *Member 4* • Appendices: source code • Tag every section with member name • Proofread, Turnitin check • Report ≤ 60 pages | All |
| **13** | **🎯 FINAL PRESENTATION + DEMO** | • Record/present final video • Live demo: open web app in browser, detect signs from webcam • Show results for all 84 images + extra images (bonus 5 marks) | All |
| **14** | **Buffer / Polish** | • Address feedback • Final cleanup and submission | All |

---

## 👥 Task Distribution (4 Members)

### Role Assignments

| Member | Role Title | Primary Responsibility | Skills to Learn |
|--------|-----------|----------------------|----------------|
| **Member 1** | **Frontend Lead** | HTML/CSS/JS frontend, webcam integration, bounding box display, responsive UI | HTML/CSS basics, JavaScript, WebRTC |
| **Member 2** | **Backend Lead** | Flask/FastAPI server, YOLO inference API, WebSocket, OpenCV processing | Flask tutorial, WebSocket, REST API |
| **Member 3** | **Data & Database Lead** | Dataset collection, annotation, augmentation, sign meaning database | Roboflow, albumentations, JSON |
| **Member 4** | **ML & Evaluation Lead** | YOLO training (Colab), model evaluation, testing, performance analysis | Ultralytics docs, Google Colab |

---

### Report Task Distribution

#### Proposal Report (30 marks, due Week 7)

| Chapter | Content | Assigned To | Marks |
|---------|---------|-------------|-------|
| **Ch 1 — Introduction** | Background, problem statement, objectives, scope, significance | **Member 1** | 3 |
| **Ch 2 — Literature Review** | Each member writes their subtopic (see below) | **All (3 pages each)** | 12 |
| **Ch 3 — Proposed Method** | System block diagram, flowchart, module descriptions, W5H | **Member 2** | 8 |
| **Ch 4 — Preliminary Work** | Each member's color/shape segmentation results | **Each separately** | 5 |
| **Ch 5 — Conclusion** | Summary, expected outcomes, timeline | **Member 3** | 2 |
| **Presentation** | 10-min video: system overview + demo | **Member 4 (coordinator), All present** | 10 |

**Chapter 4 — Preliminary Work Modules** (required by assignment):

| Member | Module | Description |
|--------|--------|-------------|
| Member 1 | Red sign segmentation using color | Convert to HSV, threshold for red hue range (H: 0-10 & 170-180, S: 70-255, V: 50-255), morphological operations, extract red sign regions |
| Member 2 | Blue sign segmentation using color | HSV threshold for blue hue range (H: 100-130, S: 50-255, V: 50-255), extract blue sign regions |
| Member 3 | Yellow sign segmentation using color | HSV threshold for yellow hue range (H: 15-35, S: 80-255, V: 80-255), extract yellow sign regions |
| Member 4 | Shape detection of signs | Canny edge detection → findContours → approxPolyDP to classify: circle (>8 vertices), triangle (3), rectangle (4), octagon (8) |

> ⚠️ Each member's code and writing sections **must be tagged with their name** — "No name no mark" policy.

---

#### Final Report (50 marks, due Week 11–13)

| Chapter | Content | Assigned To |
|---------|---------|-------------|
| **Ch 1 — Introduction** | Updated from proposal, refined objectives | **Member 1** |
| **Ch 2 — Literature Review** | Expanded with development findings | **All** |
| **Ch 3 — Methodology** | YOLO architecture, web pipeline, Flask/FastAPI architecture, frontend design | **Member 2** |
| **Ch 4 — Implementation** | Code walkthrough per module | **Each member** |
| **Ch 5 — Results & Analysis** | Recognition rates, confusion matrix, error analysis, failure conditions | **Member 3 + Member 4** |
| **Ch 6 — Conclusion & Future Work** | Summary, limitations, enhancements | **Member 4** |
| **Appendices** | Source code listings | **All** |
| **Presentation + Demo** | Final video + live demo | **Member 1 (demo), All** |

---

### Development Task Distribution

| Component | Member | Details |
|-----------|--------|---------|
| **Frontend: Webcam + Live View** | Member 1 | Browser camera access and one-at-a-time compressed HTTP frames |
| **Frontend: Detection Overlay** | Member 1 | Draw bounding boxes and labels on HTML Canvas |
| **Frontend: UI Design** | Member 1 | Spacious light theme, confidence control and detection history |
| **Backend: FastAPI Server** | Member 2 | Health endpoint and in-memory image/frame inference handler |
| **Backend: YOLO Inference** | Member 2 | Load best.pt, run inference, return JSON results |
| **Backend: OpenCV Processing** | Member 2 | Resize, color convert, draw bounding boxes for export |
| **Dataset Collection** | Member 3 | Collect at least 50 original labelled images for each of the 49 final classes |
| **Data Annotation** | Member 3 | Roboflow bounding box annotation for all images |
| **Data Augmentation** | Member 3 | Training split only: realistic rotation, scale, brightness, blur, noise and occlusion |
| **Sign Database** | Member 3 | JSON file: sign_id → name, description, category, severity |
| **YOLO Training (Colab)** | Member 4 | Train YOLO26s at 640, validate held-out metrics, export/test OpenVINO |
| **Model Evaluation** | Member 4 | Report precision, recall, F1, mAP and confusion matrix |
| **Performance Testing** | Member 4 | Measure end-to-end latency, FPS, accuracy on test set |

---

## 📚 Literature Review — Detailed Keywords & Guide

### How to Search for Papers

1. Use **Google Scholar** (scholar.google.com)
2. Filter by year: **2019–2026** (last 5–7 years)
3. Look for papers with **50+ citations** (reliable)
4. Prefer: IEEE, Springer, Elsevier, ACM, MDPI journals/conferences
5. Each member finds **5–8 papers**, picks best **3–5** to review in depth

---

### Member 1: Color-Based Road Sign Segmentation

**Search Keywords** (combine 2–3 for each search):

| Primary Keywords | Combine With |
|-----------------|-------------|
| `color segmentation traffic sign` | `HSV`, `color space`, `thresholding` |
| `red sign detection color` | `autonomous driving`, `real-time` |
| `color-based road sign detection` | `HSV threshold`, `LAB color space` |
| `traffic sign segmentation colour` | `morphological operations`, `connected components` |
| `colour filtering traffic signs` | `adaptive thresholding`, `Otsu` |

---

### Member 2: Shape Detection & Geometric Analysis for Road Signs

**Search Keywords**:

| Primary Keywords | Combine With |
|-----------------|-------------|
| `shape detection traffic sign` | `Hough Transform`, `contour analysis` |
| `geometric feature road sign` | `circle detection`, `polygon recognition` |
| `traffic sign shape classification` | `edge detection`, `Canny` |
| `contour-based sign recognition` | `approxPolyDP`, `convex hull` |
| `road sign shape segmentation` | `template matching`, `moment invariants` |

---

### Member 3: Deep Learning for Traffic Sign Recognition (YOLO, CNN)

**Search Keywords**:

| Primary Keywords | Combine With |
|-----------------|-------------|
| `deep learning traffic sign recognition` | `YOLO`, `CNN`, `real-time` |
| `YOLO26 traffic sign detection` | `end-to-end NMS-free`, `web deployment`, `OpenVINO` |
| `convolutional neural network road sign` | `transfer learning`, `fine-tuning` |
| `traffic sign detection deep learning` | `small object detection`, `data augmentation` |
| `GTSRB GTSDB benchmark` | `recognition accuracy`, `comparison` |
| `lightweight object detection` | `YOLOv5`, `SSD`, `MobileNet` |

---

### Member 4: CNN Optimization & Robustness in Adverse Conditions

**Search Keywords**:

| Primary Keywords | Combine With |
|-----------------|-------------|
| `traffic sign detection adverse conditions` | `CNN`, `YOLO`, `weather`, `lighting` |
| `region focusing traffic signs` | `parallelization`, `high-definition images` |
| `YOLO algorithm optimization` | `real-time detection`, `accuracy improvement` |
| `traffic sign recognition robustness` | `data augmentation`, `blur`, `glare` |
| `deep learning accuracy enhancement` | `feature extraction`, `preprocessing` |
| `small object detection optimization` | `efficiency`, `edge computing` |

---

### Literature Review Writing Structure (for each member)

Each member's 3-page section should follow this structure:

```
2.X [Your Subtopic Title]    (tagged: written by [Your Name])

Paragraph 1: Overview of the technique category
  - What is it? General introduction to the approach
  - Why is it relevant to traffic sign detection?

Paragraph 2-3: Paper-by-paper review
  - For each paper: Author (Year) proposed [method]. They achieved [result] on [dataset].
  - Strengths: what worked well
  - Weaknesses: limitations they mentioned or you identified
  - Compare with other papers in your section

Paragraph 4: Summary and gap analysis
  - What do existing approaches do well?
  - What gaps exist? (this is where our proposed method fills in)
  - How does this inform our system design?
```

---

## ✅ Marking Scheme Alignment Checklist

### Proposal (30 marks)

| Criteria | How We Address It |
|----------|------------------|
| Ch 1: Clear problem statement + objectives (3 marks) | Real-time traffic sign detection for driving assistance |
| Ch 2: Recent literature within 5–10 years (12 marks) | Each member reviews 3–5 papers from 2019–2026 |
| Ch 3: Novel system design with block diagram (8 marks) | YOLO + Flask/FastAPI + WebSocket + Web frontend pipeline diagram |
| Ch 4: Working preliminary segmentation (5 marks) | 4 modules: Red/Blue/Yellow color segmentation + shape detection |
| Ch 5: Clear conclusion (2 marks) | Summary + expected outcomes + timeline |
| Presentation: Clear demo video (10 marks) | Show segmentation results on test images |

### Final Project (50 marks)

| Criteria | How We Address It |
|----------|------------------|
| Well-written report following FYP2 format (20 marks) | Follow template, system diagram, W5H explanation |
| Runs within 2 seconds per image (5 marks) | Measure end-to-end YOLO26s server latency; require every acceptance run to pass the rubric rather than citing a vendor FPS estimate |
| Correct sign identification × 84 images (70 marks scaled) | Train YOLO on diverse Malaysian sign dataset |
| Beyond 84 test images (5 marks) | Test on our own collected Malaysian road sign photos |
| Report ≤ 60 pages (1 mark) | Monitor page count during writing |
| Each section tagged with member name | Enforce naming convention in every section |

---

## 🔧 Tools & Software Setup Summary

| Tool | Purpose | How to Get |
|------|---------|-----------|
| Python 3.10+ | Backend server + ML training | python.org |
| Flask or FastAPI | Web backend framework | `pip install flask` or `pip install fastapi uvicorn` |
| Flask-SocketIO | WebSocket for real-time streaming | `pip install flask-socketio` |
| Ultralytics | Train and validate YOLO26 | `pip install "ultralytics>=8.4.90,<9"` |
| OpenVINO | Intel CPU server inference | `pip install openvino` |
| OpenCV (Python) | Image processing | `pip install opencv-python` |
| VS Code | Code editor | code.visualstudio.com |
| Google Colab | Free GPU for training | colab.research.google.com |
| Roboflow | Dataset annotation (free tier) | roboflow.com |
| Git + GitHub | Version control | github.com |
| MS Visual Studio 2022 | C++ dev/testing on PC (preliminary work) | visualstudio.microsoft.com |
| Chrome/Firefox/Edge | Web app testing | Already installed |

---

## ⚠️ Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Not enough Malaysian sign images | Low accuracy | Start collecting Week 2, augment 5–10x, supplement with GTSDB dataset |
| YOLO training needs GPU | Can't train locally | Use Google Colab (free T4 GPU) — see guide above |
| Web development is new to team | Slow development | Follow Flask tutorials; start simple, iterate |
| WebSocket streaming is complex | Can't get real-time working | Fallback to image upload mode (simpler HTTP POST) |
| YOLO26s server misses latency gate | High p95 latency | Validate OpenVINO on Intel CPU; use YOLO26n only if it still passes all quality gates |
| Team member falls behind | Incomplete work | Weekly check-ins, clear ownership, buffer week 14 |
| Signs look different in real-world vs training | Poor generalization | Augment data heavily; test in various lighting conditions |

---

## 🤖 AI Agent Prompts for Team Members

> These prompts are for teammates to use with AI coding assistants (Gemini, Cursor, etc.) when building their assigned modules. **Read the full plan.md first** before using any prompt.

---

### Member 1 — Frontend Lead

#### Prompt: Webcam + Bounding Box Display
```
I am building a web frontend for a real-time traffic sign detection system.

Requirements:
- Access the user's webcam using WebRTC (navigator.mediaDevices.getUserMedia)
- Capture frames at ~10-15 FPS from the webcam video element
- Send each frame to the backend via WebSocket as base64 JPEG
- Receive detection results (bounding boxes, class labels, confidence) from the server
- Draw bounding boxes with labels on an HTML Canvas overlaying the video
- Also support image/video file upload via a file input
- Desktop/laptop layout for Chrome, Firefox and Edge; no native mobile app

Tech stack: HTML, CSS, vanilla JavaScript (no frameworks).
Create:
1. index.html - main page with webcam view, canvas overlay, upload button
2. style.css - dark theme, responsive design
3. app.js - webcam access, WebSocket connection, canvas drawing

Keep it beginner-friendly with comments explaining each step.
```

---

### Member 2 — Backend Lead

#### Prompt: Flask/FastAPI YOLO Inference Server
```
I need to build a Python web server that runs YOLO26s inference at a 640-pixel
baseline for real-time traffic sign detection.

Requirements:
- Load the validated OpenVINO export at startup on an Intel CPU server
- Keep YOLO26's default end-to-end, NMS-free detection output
- WebSocket endpoint: receive base64 JPEG frames, run inference, return JSON results
- HTTP POST endpoint: receive uploaded image file, run inference, return annotated image
- Return results as JSON: [{x, y, width, height, class_name, confidence}, ...]
- Configurable confidence threshold (default 0.5)
- Use OpenCV for image decoding/encoding
- Sign database lookup: map class_id to sign name and description

Create:
1. app.py - main Flask/FastAPI server
2. detector.py - YOLO26s/OpenVINO inference wrapper class
3. sign_database.json - sign ID to name/description mapping
4. requirements.txt - all dependencies

I'm a C++ developer new to Python web development. Explain Flask/FastAPI concepts.
```

---

### Member 3 — Data & Database Lead

#### Prompt: Dataset Preparation
```
I need to prepare a dataset of Malaysian road signs for YOLO26 object detection training.

Current situation:
- I have 84 sample images in "Color Inputs" folder (Red Signs, Blue Signs, Yellow Signs)
- I need to collect varied originals toward at least 20 images per locked class
- Signs include: speed limits, warning signs, mandatory signs, prohibitory signs
- Need to annotate with bounding boxes and class labels

Please help me with:
1. A Python script to organize and rename images into a YOLO-compatible folder structure
2. Validation of the active 63-class data.yaml for Ultralytics YOLO26 training
3. A Python script using albumentations to augment the dataset
4. Verification that the 63 canonical hyphenated names and IDs are unchanged,
   including corrected class ID 32 = pass-obstacle-on-either-side
5. Instructions on how to use Roboflow (free tier) for bounding box annotation
6. A validation split script (80% train, 10% validation, 10% test)
```

---

### Member 4 — ML & Evaluation Lead

#### Prompt: YOLO Training on Google Colab
```
I need to train a YOLO26s model on Google Colab for Malaysian road sign detection,
then deploy its validated OpenVINO export on an Intel CPU Flask/FastAPI server.

Context:
- Dataset: reviewed originals plus training-only augmentation in YOLO format
- Number of classes: exactly 49 with the canonical data.yaml order
- Dataset is uploaded to Google Drive
- YOLO26s at imgsz=640 is the baseline; YOLO26n is a latency fallback
- YOLO26m is tested only with adequate balanced data and a GPU server

Please create a complete Google Colab notebook that:
1. Installs ultralytics
2. Mounts Google Drive
3. Verifies dataset structure and data.yaml
4. Trains YOLO26s with epochs=150, imgsz=640, batch=16 and early stopping
5. Shows training metrics (mAP, loss curves)
6. Runs validation and shows confusion matrix
7. Saves best.pt and exports/validates OpenVINO against the same held-out test set
8. Reports precision, recall, mAP, p95 latency and false positives per minute

I'm new to deep learning. Please explain what each parameter means.
```

---

## ⚠️ Beginner Learning Path (First 2 Weeks)

Since the team is new to web development and deep learning, here's what each member should learn first:

| Member | Watch/Read First | Time |
|--------|-----------------|------|
| **Member 1** | [HTML & CSS Basics](https://developer.mozilla.org/en-US/docs/Learn) — MDN Web Docs | 4-6 hours |
| **Member 1** | [JavaScript Basics](https://javascript.info/) — first 5 chapters | 4 hours |
| **Member 2** | [Flask Quickstart](https://flask.palletsprojects.com/en/3.0.x/quickstart/) | 2 hours |
| **Member 2** | [Flask-SocketIO Tutorial](https://flask-socketio.readthedocs.io/) | 2 hours |
| **Member 3** | [Roboflow Annotation Tutorial](https://docs.roboflow.com/annotate) — free tier | 1 hour |
| **Member 3** | [Albumentations Tutorial](https://albumentations.ai/docs/getting_started/image_augmentation/) | 1 hour |
| **Member 4** | [Ultralytics YOLO26 Guide](https://docs.ultralytics.com/models/yolo26/) | 2 hours |
| **Member 4** | [Google Colab Intro](https://colab.research.google.com/notebooks/intro.ipynb) | 1 hour |

---

*Last updated: 2026-08-11*
*Plan version: 4.0 — YOLO26s Server-Side Web Deployment*
