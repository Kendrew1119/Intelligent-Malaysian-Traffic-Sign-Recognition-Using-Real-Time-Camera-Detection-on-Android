# ============================================
# [Member 3] split_dataset.py
# ============================================
# Module: Train/Val/Test Splitter
# Owner: Member 3 (Data & Database Lead)
#
# Purpose:
#   - Split annotated images into train/val/test sets
#   - Ratio: 80% train, 10% validation, 10% test
#   - Stratified split (proportional representation of each class)
#   - Copy images AND matching label .txt files to correct folders
#   - Print statistics: count per class per split
#
# Output Structure:
#   dataset/annotated/
#     train/images/  train/labels/
#     val/images/    val/labels/
#     test/images/   test/labels/
#
# TODO: Implement stratified dataset splitting
# ============================================
