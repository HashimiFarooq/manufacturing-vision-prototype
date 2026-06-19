import cv2
import time
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="buhQlFHUJSznlUF3heBb"
)

cap = cv2.VideoCapture(1)

last_time = 0
interval = 0.5

latest_result = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # Run inference every interval
    if now - last_time > interval:
        last_time = now

        cv2.imwrite("frame.jpg", frame)

        latest_result = client.run_workflow(
            workspace_name="heitan-kandasamy",
            workflow_id="general-segmentation-api",
            images={"image": "frame.jpg"},
            parameters={
                "classes": "Screw, Nut, Bolt, Washer, bolt_head"
            },
            use_cache=True
        )

        # 🔥 DEBUG PRINT
        print("\n--- RAW RESULT ---")
        print(latest_result)

        # Extract predictions safely
        try:
            preds = latest_result[0]["predictions"]["predictions"]

            print("\n--- DETECTIONS ---")
            for p in preds:
                print(f"{p['class']} | conf={p['confidence']:.2f}")

        except Exception as e:
            print("Parse error:", e)

    # Draw predictions
    if latest_result:
        try:
            preds = latest_result[0]["predictions"]["predictions"]

            for p in preds:
                x = int(p["x"])
                y = int(p["y"])
                w = int(p["width"])
                h = int(p["height"])

                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2

                label = f"{p['class'].strip()} {p['confidence']:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except:
            pass

    cv2.imshow("Bolt Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()