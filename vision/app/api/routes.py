from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import app.state.shared as shared 
from app.vision.stream import generate_frames


router = APIRouter()

@router.get("/detections")
def get_detections():
    return {
        "detections": shared.latest_detections
    }
    
    
@router.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
    
    
@router.get("/status")
def status():

    return {
        "camera": "connected",
        "model": "running",
        "fps": 29,
        "task_state": shared.latest_detections["task_state"]
    }