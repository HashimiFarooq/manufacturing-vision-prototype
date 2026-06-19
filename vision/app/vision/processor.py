from ultralytics import YOLO
from app.models.registry import get_model

model = get_model()
# model = YOLO("../models/yolo11n.pt")

def extract_detections(results):
    detections = {
        "task_state": "detecting",
        "timestamp": "",
        "objects": []
    }

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        detections["objects"].append({
            "label": model.names[cls],
            "center_pixel": [cx, cy],
            "confidence": round(conf, 2)
        })

    return detections


# latest_detection = {
#     "task_state": "detecting",
#     "timestamp": "2026-06-18T14:25:37Z",
#     "objects": [
#         {
#             "class": "bolt",
#             "confidence": 0.96,
#             "center": {
#                 "x": 514,
#                 "y": 276
#             }
#         },
#         {
#             "class": "washer",
#             "confidence": 0.91,
#             "center": {
#                 "x": 631,
#                 "y": 410
#             }
#         }
#     ]
# }