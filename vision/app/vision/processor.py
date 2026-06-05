from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")
model = YOLO("../models/yolo11n.pt")

def extract_detections(results):
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        detections.append({
            "label": model.names[cls],
            "center_pixel": [cx, cy],
            "confidence": round(conf, 2)
        })

    return detections