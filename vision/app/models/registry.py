from ultralytics import YOLO

MODEL_PATHS = {
    "default": "/app/models/yolo11n.pt",
    "latest_test": "/app/models/latest_test.pt",
    "tool_detector": "/app/models/tool_model.pt"
}

_models_cache = {}

def get_model(name="default"):
    if name not in _models_cache:
        _models_cache[name] = YOLO(MODEL_PATHS[name])
    return _models_cache[name]