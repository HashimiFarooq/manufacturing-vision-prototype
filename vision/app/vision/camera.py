import cv2
from ultralytics import YOLO
from app.vision.detector import run_inference
from app.vision.processor import extract_detections
from app.state.shared import latest_detections, latest_frame

model = YOLO("models/yolo11n.pt")


def start_camera():
    # Open camera
    cam_index = 1
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = run_inference(frame)

        detections = extract_detections(results)

        # ---- DRAWING LAYER (camera.py responsibility) ----
        annotated_frame = results[0].plot()

        for d in detections:
            cx, cy = d["center_pixel"]

            cv2.circle(
                annotated_frame,
                (cx, cy),
                10,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                annotated_frame,
                f'{d["label"]} {d["confidence"]:.2f}',
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # ---- UPDATE SHARED STATE ----
        latest_detections.clear()
        latest_detections.extend(detections)

        global latest_frame
        latest_frame = annotated_frame