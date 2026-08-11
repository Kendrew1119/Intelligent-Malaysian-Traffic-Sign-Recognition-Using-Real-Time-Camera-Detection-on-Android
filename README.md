# MYSignVoice

MYSignVoice is a web-based Malaysian traffic-sign detection and classification system. A browser supplies a live webcam stream or an uploaded image/video; a Python backend returns bounding boxes, one of 49 sign classes, confidence scores, and a temporally stable result for display or speech output.

## Active design

| Component | Technology | Purpose |
|---|---|---|
| Web client | Browser camera (`getUserMedia`) and WebSocket/HTTP | Capture frames and render results |
| Backend | Python with Flask or FastAPI | Load the model once and serve inference |
| Detector | Ultralytics YOLO26s, pretrained and fine-tuned at `imgsz=640` | Final 49-class detection and classification |
| CPU deployment | OpenVINO on the Intel laptop/server | Lower-latency server inference |
| Optional candidate path | OpenCV HSV masks, morphology, and contours | Propose contextual crops for a second YOLO pass |
| Training | Roboflow annotations and Google Colab | Dataset versioning, training, validation, and export |

YOLO26s is the primary model. YOLO26n is a measured fallback for a slow or highly concurrent CPU server; YOLO26m is only a challenger for a suitable NVIDIA/cloud GPU. Model size is selected from validation accuracy and end-to-end latency on the actual deployment machine, not from model-family benchmarks alone.

The OpenCV branch is optional. Periodic full-frame YOLO inference remains the recall-preserving path, so a failed colour mask cannot hide a sign. Full-frame and ROI results are merged, then a class must remain consistent across frames before the system displays or announces it.

## Current status

- The 49-class contract is locked in `dataset/data.yaml`.
- The preliminary OpenCV code demonstrates colour segmentation, contour filtering, broad shape detection, and a hybrid webcam prototype.
- The Colab script is prepared for YOLO26s training and ONNX/OpenVINO export.
- A canonical trained 49-class `best.pt` does not exist yet. The preliminary `best.pt` is test-only and must not be deployed as the final model.
- The web frontend/backend still needs to be implemented after the first validated model is available.

## Project structure

```text
miniproject/
├── dataset/                         # Canonical 49-class config and local data folders
├── training/                        # Colab training, evaluation, and export utilities
├── preliminary/                     # OpenCV and hybrid webcam experiments
├── docs/                            # Dataset, training, design, and report guides
├── enhanced_block_diagram.md        # Active end-to-end system design
├── member4literature_review.md      # Literature synthesis and proposed pipeline
└── plan.md                          # Active web-project plan
```

Older Android/ncnn documents and conversion helpers are archival material from the previous project direction. They are not part of the active server-side web pipeline.

## Next phase

1. Annotate the images in Roboflow with the exact 49 dash-separated class names from `dataset/data.yaml`.
2. Add diverse real camera scenes and genuine no-sign/hard-negative images; keep train, validation, and test scenes independent.
3. Run `training/train_colab.py` in Google Colab with the default `yolo26s.pt` weights.
4. Evaluate per-class recall, mAP50-95, false detections on no-sign footage, and p50/p95 end-to-end latency.
5. Benchmark the full-frame and optional hybrid modes on the target laptop before enabling ROI inference in the web backend.

See `docs/roboflow_training_guide.md` for the exact workflow.

## License note

This repository is an academic project. Before public or commercial web deployment, review the Ultralytics model/code licence and choose an AGPL-compliant open-source release or the appropriate commercial licence.
