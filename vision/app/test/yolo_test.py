from ultralytics import YOLO
import cv2

# Load model
model = YOLO("../models/yolo11n_v3.pt")

# Run inference
results = model("../static/nails_and_screws.jpg")

# Get annotated image (with boxes drawn)
annotated_frame = results[0].plot()
print(f"Image shape: {annotated_frame.shape}")
# Print detection results
for r in results:
    boxes = r.boxes

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0]
        conf = b.conf[0]
        cls = int(b.cls[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        cv2.circle(annotated_frame, (cx, cy), 10, (0, 0, 255), -1)

        print(f"Class: {cls}, Confidence: {conf:.2f}, Center: ({cx:.1f}, {cy:.1f})")


cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 1280, 720)

# Show image
cv2.imshow("YOLO Detection", annotated_frame)


# Keeps window open until you press a key
cv2.waitKey(0)
cv2.destroyAllWindows()