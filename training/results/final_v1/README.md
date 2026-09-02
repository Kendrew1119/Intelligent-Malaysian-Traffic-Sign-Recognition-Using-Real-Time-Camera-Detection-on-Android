# Final YOLO26s training results

These artifacts document the completed 63-class baseline, the controlled
`cls_pw=0.25` tuning run, their validation comparison, and the single final
evaluation on the untouched test split.

Selected checkpoint: `models/best.pt` (tuned YOLO26s).

Final test metrics:

- Precision: 0.9616
- Recall: 0.8692
- F1: 0.9131
- mAP50: 0.9484
- mAP50-95: 0.8043

The test set contains 782 images and 811 annotated instances. Several rare
classes have too few test examples to support a strong per-class reliability
claim, even though aggregate performance is high.
