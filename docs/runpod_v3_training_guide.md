# MYSignVoice Version 3 export and RunPod training

This guide matches `training/train_yolo26_v3_runpod.ipynb`.

## Do the 367 new images need a split?

Yes. Do not put all 367 images into Train.

The batch was added to Roboflow as:

| Split | New images |
|---|---:|
| Train | 261 |
| Validation | 51 |
| Test | 55 |
| Total | 367 |

Before Version 3 generation, the complete project contained:

| Split | All images |
|---|---:|
| Train | 6,291 |
| Validation | 1,618 |
| Test | 838 |
| Total | 8,747 |

These camera screenshots were captured consecutively. A normal random split can
put nearly identical frames in different splits and make the score look better
than real camera performance. The notebook therefore regroups the 367 images by
timestamp and label. Frames with the same labels and no more than eight seconds
between them remain in one split. The notebook prints the final leakage-safe
counts. It does not move the old Version 2 images.

## Part A — Export Roboflow Version 3

Version 3 has already been started with these settings:

- name: `MYSignVoice V3 Camera Retraining`;
- 8,747 source images;
- 63 classes;
- Auto-Orient enabled;
- Resize set to **Fit within 640 x 640**;
- Roboflow augmentation turned off;
- 367 new images confirmed in the completed annotation job.

When generation finishes:

1. Open **Versions** in the MYSignVoice project.
2. Open **v3 — MYSignVoice V3 Camera Retraining**.
3. Click **Download Dataset**.
4. Select **YOLO26**. If Roboflow only shows **YOLOv8**, select it; the label
   layout is compatible with Ultralytics YOLO26 training.
5. Choose **Download ZIP to Computer**.
6. Rename the downloaded file to `mysignvoice_v3_yolo26.zip`.

Do not add another Roboflow augmentation step and do not generate another
version unless the v3 export is genuinely incorrect.

## Part B — Start a RunPod pod

1. Open RunPod and deploy a GPU pod with JupyterLab.
2. An RTX A4000/A4500/A5000, RTX 3090, RTX 4090 or similar GPU is sufficient.
   At least 12 GB GPU memory is recommended.
3. Use a persistent volume if available so a temporary browser disconnection
   does not remove the training files.
4. Open **JupyterLab** after the pod becomes ready.

## Part C — Fast API download (recommended)

Upload these two files into `/workspace` using the JupyterLab file browser:

1. `training/train_yolo26_v3_runpod.ipynb`;
2. the current local `models/best.pt` renamed to `best_v2.pt`.

Leave `USE_ROBOFLOW_API = True` in Cell 2. Cell 3 displays a hidden password
prompt. Paste the private Roboflow API key and press Enter. RunPod then downloads
and extracts Version 3 directly from Roboflow, which is normally much faster
than uploading the 276 MB ZIP through JupyterLab.

The key is not displayed or written into the notebook. Keep it out of normal
code cells, screenshots, Git and reports. If API download is unavailable, set
`USE_ROBOFLOW_API = False` and upload `mysignvoice_v3_yolo26.zip` manually.

## Part D — Run the notebook

Open `train_yolo26_v3_runpod.ipynb` and run the cells in order.

1. **Cell 1** installs the tested packages.
2. **Cell 2** confirms the GPU, V2 checkpoint and selected dataset source.
3. **Cell 3** asks for the hidden API key, downloads Version 3 and verifies that
   its 63 class IDs exactly match the deployed Version 2 checkpoint.
4. **Cell 4** regroups consecutive camera frames so a burst cannot leak across
   Train, Validation and Test.
5. **Cell 5** checks label syntax, class coverage, exact duplicates and split
   leakage, then creates dataset graphs for the report.
6. **Cell 6** measures the existing V2 model on the new Validation split.
7. **Cell 7** fine-tunes YOLO26s from `best_v2.pt` for up to 60 epochs. Early
   stopping can finish it sooner.
8. **Cell 8** compares V2 and V3 on Validation and creates overall and per-class
   results.
9. Leave **Cell 9** with `RUN_FINAL_TEST = False` until the Validation comparison
   is accepted. Then change it to `True` in Cell 2, rerun Cell 2, and run Cell 9
   once.
10. **Cell 10** exports PyTorch, ONNX and OpenVINO files and creates a result
    package.
11. **Cell 11** shows the ZIP download link.

Do not run several cells at the same time. The number beside a Jupyter cell is
its execution order. `[*]` means that cell is still running. During Cell 7, the
output table updates once per epoch and shows GPU memory, losses and progress.

## Part E — Decide whether V3 replaces V2

Do not replace the current web-app model only because training completed.

Use Validation first. Prefer V3 when:

- mAP50-95 and the precision-recall balance are not materially worse;
- the newly collected camera classes improve;
- the seven speed-limit class rows improve or remain stable;
- the camera false-positive safeguards still work.

A small overall change can be acceptable when camera-class performance clearly
improves, but record the trade-off. If global mAP50-95 falls by more than about
0.01, keep V2 and investigate the new annotations or sampling balance before
deployment.

After the settings are frozen, run the Test cell once. Then download the result
ZIP and verify it exists on the computer before stopping or terminating the pod.

## Important model decision

This notebook retrains the full 63-class YOLO26s detector using all 8,747
images. It does not train only on the 367 new images. The existing V2 `best.pt`
is the starting checkpoint, which avoids discarding what the model already
learned.

The optional speed-limit number reader remains a separate experiment. First
evaluate this stronger V3 detector. If numbers such as 30 and 80 are still
confused after V3, retrain the auxiliary seven-number reader using crops from
the frozen V3 export rather than replacing YOLO with generic OCR.
