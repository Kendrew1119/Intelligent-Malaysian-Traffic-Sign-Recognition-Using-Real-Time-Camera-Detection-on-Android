# ============================================
# [Member 3] organize_dataset.py
# ============================================
# Module: Dataset Organizer
# Owner: Member 3 (Data & Database Lead)
#
# Purpose:
#   - Organize raw road sign images into YOLO-compatible folder structure:
#       dataset/
#         annotated/
#           train/images/  train/labels/
#           val/images/    val/labels/
#           test/images/   test/labels/
#   - Rename images to consistent format: sign_{classid}_{index}.png
#   - Copy the 84 provided "Color Inputs" images into raw_images/
#   - Split: 80% train, 10% val, 10% test
#
# YOLO Label Format (each .txt file, one line per object):
#   class_id center_x center_y width height
#   (all values normalized 0-1)
#
# References:
#   - Ultralytics dataset format: https://docs.ultralytics.com/datasets/detect/
#   - See plan.md → Member 3 Prompt: Dataset Preparation
#
# TODO: Implement dataset organization script
# ============================================
