# MYSignVoice annotated update: split, export, and train

Use this guide after the new 300+ images have been annotated and reviewed in
Roboflow. The matching notebook is
`training/train_yolo26_annotated_update.ipynb`.

## Short answer: do the 300+ images need a split?

Yes. Split the **original images before any augmentation**.

For 300 independent images, a useful starting point is:

| Split | Share | About 300 images | Purpose |
|---|---:|---:|---|
| Train | 70% | 210 | Teaches the model |
| Validation | 20% | 60 | Selects settings and the best checkpoint |
| Test | 10% | 30 | Used once, after all choices are frozen |

The exact count can be a little different. Class coverage and source separation
matter more than getting exactly 210/60/30.

### Important rule for camera or video frames

Do not randomly split neighbouring frames. Images from the same video, camera
session, printed sign, road location, or burst of photos are related. Put the
whole group in only one split. Otherwise the model may see almost the same image
in training and testing, making the score look better than real performance.

Examples:

- `session_01` -> Train only
- `session_02` -> Validation only
- `session_03` -> Test only

If these 300+ images are being added to the existing Version 2 project, do not
move the old held-out test images into Train. Keep their split fixed and assign
the new source groups around 70/20/10. If a class has too few independent source
groups to appear in all three splits, do not duplicate it. Put more examples in
Train, record the limitation, and collect another independent group later.

## Part A - Check the annotations

1. Open the MYSignVoice object-detection project in Roboflow.
2. Open **Annotate** and confirm the new batches are in **Dataset**, not
   **Unassigned** or **Annotating**.
3. Review every bounding box:
   - the box covers the complete traffic-sign face;
   - the box does not contain a large area of sky, pole, road, or wall;
   - the selected class is correct;
   - every visible target sign is labelled;
   - an image with no target sign is marked as a null/negative image, not left
     unfinished.
4. Keep the same 63 classes. Do not rename, merge, add, or reorder classes.
5. Use batch names or tags to identify related sessions before splitting.

## Part B - Create the split and frozen Roboflow version

1. Open **Dataset** and inspect the Train, Valid, and Test counts.
2. Keep each related session in one split only.
3. Aim for about **70% Train, 20% Valid, and 10% Test** for the 300+ new
   originals.
4. Open **Versions** in the left menu.
5. Click **Generate New Version**.
6. At **Train/Test Split**, verify the totals. Use **Rebalance** only when the
   images are genuinely independent. Do not let rebalancing scatter adjacent
   frames from one session across several splits.
7. Preprocessing:
   - enable **Auto-Orient**;
   - resize to **640 x 640**;
   - choose an aspect-preserving **Fit** option, not Stretch.
8. For this first controlled update, select **no Roboflow augmentation** and a
   maximum version size of **1x**. The notebook applies safe training-only
   augmentation. This avoids accidental double augmentation.
9. Give the version a clear name, for example
   `v3-300-annotated-camera-update`.
10. Click **Generate** and wait until the frozen version is ready.
11. Save these details in your notes:
    - Roboflow version number;
    - Train image count;
    - Valid image count;
    - Test image count;
    - total image count;
    - one screenshot of the version summary.

Roboflow documents Versions as frozen snapshots and provides the split controls
under **Versions -> Generate New Version**:
<https://docs.roboflow.com/datasets/dataset-versions/create-a-dataset-version>

## Part C - Export the dataset

You do not need to download and upload a large ZIP manually. The notebook uses
the Roboflow Python package to download the selected frozen version.

1. Open the generated version.
2. Click **Export** or **Download Dataset**.
3. Select **YOLO26** if it appears. If it does not appear, select **YOLOv8**;
   both use the Ultralytics detection image/label layout required here.
4. Select **Show Download Code**.
5. Copy only these identifiers into your private notes:
   - workspace slug;
   - project slug;
   - version number.
6. Do not copy the private API key into Git, a notebook cell, a report, or a
   screenshot.

Roboflow's export documentation explains that a generated version can be
downloaded through the web interface or Python package:
<https://docs.roboflow.com/datasets/dataset-versions/exporting-data>

## Part D - Prepare Google Colab

1. Open <https://colab.research.google.com/>.
2. Upload `training/train_yolo26_annotated_update.ipynb`.
3. Select **Runtime -> Change runtime type -> T4 GPU**.
4. Open Colab **Secrets** using the key icon.
5. Add a secret named `ROBOFLOW_API_KEY`.
6. Paste the private Roboflow API key as the secret value.
7. Enable notebook access for that secret.
8. Upload the current deployed Version 2 `models/best.pt` when the notebook asks
   for it. This is the starting checkpoint; do not start again from a generic
   model unless a separate baseline experiment is intended.

## Part E - Run the notebook safely

1. Run each cell from top to bottom.
2. In the configuration cell, enter the correct workspace, project, and new
   version number.
3. Leave `RUN_FINAL_TEST = False`.
4. Run the dataset download and audit cells.
5. Compare the printed split counts with the Roboflow version-summary screenshot.
6. Copy the printed counts into `EXPECTED_SPLIT_IMAGES`.
7. Set `SPLIT_APPROVED = True` only after checking:
   - all 63 class IDs match the deployed model;
   - Train, Valid, and Test exist;
   - there are no exact duplicate images across splits;
   - related sessions were not separated;
   - Test was not used to choose settings.
8. Rerun the configuration and audit cells.
9. Run baseline validation, fine-tuning, and validation comparison.
10. Choose the model using Validation results only.
11. When the team agrees that no more tuning will happen, set
    `RUN_FINAL_TEST = True` and run the final-test cell once.
12. Run the export cell. It creates:
    - `best.pt` for training/reference;
    - `best.onnx` for portable inference;
    - `best_openvino_model/` for the Intel CPU web server;
    - metrics, plots, dataset summary, manifest, and a ZIP backup in Drive.

Ultralytics supports training from a pretrained checkpoint and exporting custom
models to ONNX and OpenVINO:
<https://docs.ultralytics.com/modes/train>
<https://docs.ultralytics.com/modes/export>

## What not to do

- Do not put all 300+ images into Train.
- Do not split adjacent frames randomly.
- Do not augment Validation or Test.
- Do not use horizontal or vertical flips; they change left/right sign meaning.
- Do not change the 63-class order.
- Do not paste the Roboflow API key into the notebook.
- Do not check Test repeatedly while changing epochs, confidence, class weights,
  or augmentations.
- Do not replace the deployed model until the new checkpoint and OpenVINO export
  pass the same validation, final-test, and saved-camera checks.
