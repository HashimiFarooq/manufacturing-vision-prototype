from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")

def run_inference(frame):
    return model(frame, verbose=False)