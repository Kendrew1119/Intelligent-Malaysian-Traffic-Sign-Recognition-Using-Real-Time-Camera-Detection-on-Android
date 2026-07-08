import cv2
from ultralytics import YOLO

# 1. Load the trained model
# Note: You need to download 'best.pt' from your Google Drive into the dataset/ folder!
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "dataset", "best.pt") 

try:
    # Try to load the PyTorch model
    model = YOLO(model_path)
    print(f"✅ Successfully loaded model: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n⚠️ ACTION REQUIRED: You need to download 'best.pt' from your Google Drive!")
    print("Go to: TrafficSignProject/training_runs/mtsd_v1/weights/best.pt")
    print("Download it and place it inside the 'dataset' folder on your Desktop.")
    exit()

# 2. Open the laptop webcam (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("\n📷 Webcam opened! Hold up a traffic sign to the camera.")
print("Press 'q' to quit the window.\n")

# 3. Read frames and run detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to grab frame.")
        break

    # Run YOLOv8 inference on the frame
    # conf=0.4 means it will only show detections with > 40% confidence
    results = model.predict(source=frame, conf=0.4, verbose=False)

    # The results object has a built-in method to draw the bounding boxes
    annotated_frame = results[0].plot()

    # Display the resulting frame on your laptop screen
    cv2.imshow('YOLOv8 Traffic Sign Detection - Live Test', annotated_frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Clean up
cap.release()
cv2.destroyAllWindows()
