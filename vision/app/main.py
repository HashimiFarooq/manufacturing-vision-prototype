from fastapi import FastAPI
from app.api.routes import router
from app.vision.camera import start_camera
import threading

app = FastAPI()
app.include_router(router)

# start YOLO in background thread
threading.Thread(target=start_camera, daemon=True).start()