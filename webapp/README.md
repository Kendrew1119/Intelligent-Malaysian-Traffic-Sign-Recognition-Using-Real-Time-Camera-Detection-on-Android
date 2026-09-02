# MYSignVoice local web app

The app serves the browser interface and the trained OpenVINO model from the
same laptop. Images are processed in memory and are not saved by the server.

## Start

From the repository root in PowerShell:

```powershell
python -m pip install -r requirements-web.txt
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Select **Upload image** or **Live camera**.
The browser will ask for camera permission the first time live camera is used.

When a sign is missed or classified incorrectly, select **Save difficult
frame**, choose the issue and correct class, then save it. The original frame
is stored in `dataset/hard_cases/images/` and its review details are appended to
`dataset/hard_cases/manifest.jsonl`. Nothing is uploaded automatically.

## Runtime behaviour

- Loads `models/best_openvino_model` once when the server starts.
- Runs full-frame YOLO26s inference at 640 pixels.
- Uses YOLO26 Version 3 to locate and classify speed-limit signs, then runs a
  lightweight offline OCR recognizer only on the proposed sign crop. OCR can
  report plausible values from 5 to 130 in five-unit steps. A different OCR
  value overrides YOLO only at 95% OCR confidence; otherwise YOLO remains the
  fallback. OCR failure never produces "number unclear" or blocks speech. The
  low-coverage experimental SpeedLimitCNN remains disabled.
- Captures the browser camera at its preferred 1280 x 720 resolution, then
  submits an aspect-preserving frame capped at 960 x 540 every 400 ms. A new
  request is never started while the previous request is still running.
- Uses a 20% default for uploaded images and a safer 35% minimum for the live
  camera. Camera predictions from 20% to below the selected threshold are shown
  as uncertain but are not spoken.
- Ignores camera boxes that are too small to retain reliable sign detail.
- Tracks the same class using box overlap, centre movement and size change, so
  normal camera or vehicle movement does not require a static bounding box.
- Allows one missed processed frame and expires a track after 1.8 seconds.
- Confirms after two spatially matching detections when both are at least 60%
  confidence; otherwise it requires three matching detections. Confirmed speech
  retains the five-second per-class cooldown.
- Shows a plain-language meaning and recommended response for every supported
  sign, based on the Malaysian JKR traffic-sign categories.
- Does not upload images to Roboflow or any external service.
