# ============================================
# [Member 4] train_colab.py
# ============================================
# Module: YOLOv8-nano Training Script (for Google Colab)
# Owner: Member 4 (ML & Evaluation Lead)
#
# Purpose:
#   - Train YOLOv8-nano on Malaysian road sign dataset
#   - Designed to run on Google Colab with free T4 GPU
#   - Steps:
#       1. Install ultralytics
#       2. Mount Google Drive
#       3. Load dataset from Drive
#       4. Train YOLOv8n with:
#           epochs=100, imgsz=640, batch=16, patience=20
#       5. Save best model to Drive
#       6. Show training metrics (mAP, loss curves)
#       7. Run validation and show confusion matrix
#
# Usage (in Colab):
#   !python train_colab.py
#   OR copy cells into Colab notebook
#
# References:
#   - Ultralytics docs: https://docs.ultralytics.com/
#   - See plan.md → Google Colab Training Guide
#   - See plan.md → Member 4 Prompt: YOLO Training on Google Colab
#
# TODO: Implement training script
# ============================================
