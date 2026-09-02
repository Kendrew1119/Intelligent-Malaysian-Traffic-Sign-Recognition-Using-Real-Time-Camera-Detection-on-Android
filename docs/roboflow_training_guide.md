# MYSignVoice - Google Colab Training Guide (63 Classes)

This is the active server-side web deployment workflow. It trains the locked
63-class inventory in `dataset/data.yaml` with **YOLO26s at `imgsz=640`** as
the baseline. The training script stops before training if the exported Roboflow
names or IDs differ from the canonical list.

The class mapping is unchanged by the model migration. IDs are zero-based, and
corrected class ID 32 must remain `pass-obstacle-on-either-side`.

> **Deployment scope:** there is no Android or iOS application in the active plan.
> `docs/android_app_guide.md` and `training/convert_to_ncnn.py` are archival
> experiments only; do not use them for the current build or acceptance test.

## 1. Create the Colab notebook

1. Open Google Colab and select **Runtime -> Change runtime type -> T4 GPU**.
2. Clone/copy the repository to `/content/miniproject`, or upload
   `training/train_colab.py`.
3. Keep the Roboflow API key in a Colab secret or environment variable. Never put
   it in a Git-tracked file, notebook output, or screenshot.

## 2. Install packages and mount Drive

```python
!pip install -q "ultralytics>=8.4.90,<9" roboflow pyyaml openvino onnx onnxruntime

from google.colab import drive, userdata
drive.mount("/content/drive")

import os
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
```

Create the `ROBOFLOW_API_KEY` entry in Colab's **Secrets** panel first and grant
the notebook access. Do not paste the key directly into a code cell.

## 3. Get the Roboflow identifiers

Open the generated Roboflow dataset version and choose **Download Dataset -> Show
download code**. Copy the workspace slug, project slug, and version number:

```python
# rf.workspace("your-workspace").project("mysignvoice-49-signs").version(1)
```

Roboflow may call the compatible export option **YOLOv8**. That name describes the
standard Ultralytics YOLO text-label layout; YOLO26 uses the same labels. Do not
remap IDs by hand. Generate a corrected Roboflow version if its `data.yaml` does
not exactly match `dataset/data.yaml`.

## 4. Train the 640 baseline

```python
%cd /content/miniproject/training

!python train_colab.py \
  --workspace "your-workspace" \
  --project "mysignvoice-49-signs" \
  --version 1 \
  --model "yolo26s.pt" \
  --run-name "yolo26s_640_rf_v1" \
  --imgsz 640 \
  --epochs 150 \
  --batch 16
```

Use batch 8 if the Colab GPU runs out of memory. Keep `imgsz=640` for the first
controlled run. A higher input size is a later challenger and must be tested on the
same split and deployment hardware.

The script downloads data to Colab local storage, validates all 63 class names,
trains the model, exports ONNX and OpenVINO, and copies the run to:

```text
/content/drive/MyDrive/TrafficSignProject/training_runs/yolo26s_640_rf_v1/
```

Retain `weights/best.pt`, the OpenVINO model directory, metrics, plots, arguments,
and the tested `data.yaml` together so a result can be reproduced.

## 5. Use the default YOLO26 inference path

YOLO26 detection uses its one-to-one, end-to-end head by default. It outputs final
detections directly, so the normal Ultralytics prediction and export path is
**NMS-free**. Do not add external detector NMS or set `end2end=False` in the
baseline.

The safe-hybrid application may still use class-aware IoU to merge a full-frame
result with a crop result. That is cross-pass application deduplication, not
traditional detector NMS. A traditional one-to-many/NMS export is allowed only as
a documented compatibility experiment and must be benchmarked separately.

See the official [YOLO26 model guide](https://docs.ultralytics.com/models/yolo26/),
[end-to-end detection guide](https://docs.ultralytics.com/guides/end2end-detection/),
and [OpenVINO integration](https://docs.ultralytics.com/integrations/openvino/).

## 6. Validate the Intel CPU deployment

For the target Intel CPU server, use the exported OpenVINO model as the deployment
candidate. Validate `best.pt` and its OpenVINO export on the same held-out images
before webcam testing. Do not assume an export is equivalent merely because it
loads.

The deployment run must record:

- precision, recall, mAP@0.5, mAP@0.5:0.95, per-class recall, and confusion matrix;
- median and 95th-percentile end-to-end latency at 640;
- processed FPS, small/distant-sign recall, false positives per minute on no-sign
  video, and unstable label changes; and
- PyTorch-to-OpenVINO metric difference on the identical test set.

## 7. Model and pipeline acceptance gates

Freeze the test split before comparing models. Use the same 640 input, validation
thresholds, saved laptop-camera recordings, server, and measurement method.

| Decision | Measured gate |
|---|---|
| Accept YOLO26s baseline | Precision and recall are each at least 0.80, mAP@0.5 is at least 0.75, and target-server p95 end-to-end latency is at most 500 ms on Intel CPU or 200 ms on the selected GPU server. |
| Accept OpenVINO export | It passes the same functional tests and loses no more than 0.01 absolute mAP@0.5 versus its `best.pt` source; report latency rather than assuming a speedup. |
| Use YOLO26n fallback | YOLO26s misses the latency gate, and YOLO26n passes both the latency gate and all minimum quality gates. |
| Test YOLO26m challenger | Only after all 63 classes have adequate, balanced original data and a GPU server is available. Accept it only if it improves mAP@0.5:0.95 or small-sign recall by at least 0.02 absolute without failing the latency or false-positive gate. |
| Enable safe hybrid | Against full-frame YOLO26s on identical recordings, it must improve p95 latency by at least 10% or small-sign recall by at least 0.02 absolute, while reducing no overall recall by more than 0.01 and not increasing false positives per minute. |

Thresholds are selected on validation data, then frozen before the final test. If
the seed dataset cannot support each class in every split, label the run as a demo;
do not claim that it satisfies the final quality gates.

## 8. Improve data before architecture

The 84-image seed set is not sufficient evidence for a 63-class production model.
First add varied originals, laptop-camera examples, hard negatives, and realistic
training-only blur/noise/compression augmentation. Do not use horizontal or vertical
flips.

YOLO26n is a deployment fallback, not the accuracy baseline. YOLO26m is not a
default upgrade. Do not add a custom attention block, detection head, or backbone
until the standard YOLO26s error analysis shows a reproducible limitation.
