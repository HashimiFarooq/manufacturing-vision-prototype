from ultralytics import YOLO
from app.models.registry import get_model

model = get_model()
def run_inference(frame):
    return model(frame, verbose=False)