# Active MYSignVoice model

The local canonical detector is the 63-class YOLO26s Version 3 checkpoint. It
was fine-tuned from the deployed Version 2 checkpoint on the complete,
leak-audited Roboflow Version 3 dataset. Exact duplicate images found across
splits were removed before training.

- `best.pt`: master PyTorch checkpoint used for validation and future export.
- `best.onnx`: portable web/backend deployment export.
- `best_openvino_model/`: Intel CPU deployment export.
- `data.yaml`: exact Roboflow alphabetical class-ID order embedded in the model.
- `model_manifest.json`: selected training configuration and untouched test metrics.
- `speed_limit_reader.pt`: archived experimental second-stage reader for the
  seven supported speed values. It is not used by the active web application.
- `speed_limit_reader_manifest.json`: reader class order, crop size, confidence
  gate, and prototype evaluation results.

The model files are intentionally excluded from Git because they are generated
binary artifacts. The complete Version 1 model remains preserved under
`mysignvoice_yolo26s_63class_final_v1/`, and the previous deployed Version 2
model is preserved under `mysignvoice_yolo26s_63class_v2/`. Keep an external
backup of the Version 3 checkpoint as well. Runtime
code must use `model.names` (or this model-specific YAML) for class IDs. The
older `dataset/data.yaml` contains the reviewed seed inventory order and has the
same 63-name set, but its numeric order differs from the trained Roboflow export.

The detector retains all seven original speed-limit classes. The web app uses
them as speed-sign proposals and applies RapidOCR recognition only inside each
proposed box. The OCR path accepts values from 5 to 130 in five-unit steps and
requires 95% OCR confidence before a value different from YOLO can replace the
numeric class. A failed or weak OCR result falls back to YOLO and never blocks
speech. The archived SpeedLimitCNN remains disabled because it accepted only
one third of its small test set and frequently returned "number unclear" on
laptop-camera crops. The ordinary camera confidence, readable-size and
multi-frame safeguards still apply before any speed-limit announcement.

Version 3 test results on 838 images and 867 instances: precision 0.9304,
recall 0.8670, F1 0.8976, mAP50 0.9386, and mAP50-95 0.7770. On the same
Version 3 test split, it improved over Version 2 by 1.29 percentage points in
precision, 0.42 points in mAP50 and 0.84 points in mAP50-95, while recall fell
by 2.27 points. The reviewed 84-image top-class check improved from 83/84 with
two additional wrong low-confidence detections to 84/84 with none at a 20%
threshold. This fixed-set result is not bounding-box mAP, and rare classes with
only one or a few test instances are not established as universally reliable.
