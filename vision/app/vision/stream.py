import time
import cv2
import app.state.shared as shared

def generate_frames():
    while True:

        if shared.latest_frame is None:
            time.sleep(0.01)
            continue

        frame = shared.latest_frame.copy()

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )