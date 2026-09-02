# Speed-limit reader dataset

This auxiliary dataset was generated from the labelled speed-limit boxes in
the Roboflow Version 2 YOLO export. The seven original detector classes were
not merged or deleted.

- `summary.json` records crop counts by split and numeric value.
- `manifest.csv` connects each crop to its source image, class, and split.
- `crops/` is generated locally and excluded from Git.

Rebuild the crops from a downloaded YOLO export with:

```powershell
python training\prepare_speed_limit_reader.py "C:\path\to\roboflow-export.zip" --replace
```

Train the auxiliary reader with:

```powershell
python training\train_speed_limit_reader.py
```

The current test split contains only 24 speed-limit instances and several
values have one to three test examples. Its results are therefore preliminary;
use an independent camera test set before claiming real-world number-reading
accuracy.
