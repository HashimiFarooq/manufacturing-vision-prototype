import cv2
from app.state.shared import latest_frame

def generate_frames():
    while True:
        if latest_frame is None:
            continue

        # encode frame as JPEG
        success, buffer = cv2.imencode('.jpg', latest_frame)
        if not success:
            continue

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )