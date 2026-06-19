import cv2
from ultralytics import YOLO
from app.vision.detector import run_inference
from app.vision.processor import extract_detections
import app.state.shared as shared
from app.models.registry import get_model

model = get_model()

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
        
        for d in detections["objects"]:
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
        # shared.latest_detections.clear()
        # shared.latest_detections.extend(detections)
        
        shared.latest_detections = detections
        shared.latest_frame = frame.copy()