from ultralytics import YOLO
import cv2

# Load model
model = YOLO("../models/yolo11n.pt")

# Open camera
cam_index = 1
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO on current frame
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        conf = float(box.conf[0])
        cls = int(box.cls[0])

        print({
            "label": model.names[cls],
            "center_pixel": [cx, cy],
            "confidence": round(conf, 2)
        })

        cv2.circle(
            annotated_frame,
            (cx, cy),
            10,
            (0, 0, 255),
            -1
        )

        cv2.imshow("YOLO Live Detection", annotated_frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()