from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.state.shared import latest_detections
from app.vision.stream import generate_frames


router = APIRouter()

@router.get("/detections")
def get_detections():
    return {
        "detections": latest_detections
    }
    
    
@router.get("/frame")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )