# ============================================
# [Member 3] augment_dataset.py
# ============================================
# Module: Data Augmentation
# Owner: Member 3 (Data & Database Lead)
#
# Purpose:
#   - Augment training images to multiply dataset 5-10x
#   - Uses albumentations library
#   - Augmentations to apply:
#       - HorizontalFlip (p=0.5)
#       - RandomBrightnessContrast (p=0.5)
#       - GaussNoise (p=0.3)
#       - MotionBlur (p=0.2)
#       - RandomRain / RandomFog (p=0.1) — simulate weather
#       - Rotate (limit=15 degrees, p=0.3)
#       - ColorJitter (p=0.3) — simulate different lighting
#   - Must also transform bounding box annotations accordingly
#   - Save augmented images + labels in YOLO format
#
# Install:
#   pip install albumentations
#
# References:
#   - albumentations docs: https://albumentations.ai/docs/
#   - See plan.md → Member 3 Prompt: Dataset Preparation
#
# TODO: Implement augmentation pipeline
# ============================================
