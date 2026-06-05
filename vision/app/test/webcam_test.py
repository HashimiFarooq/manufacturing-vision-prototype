import cv2

cam_index = 1
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(
        frame,
        f"Camera Index: {cam_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()