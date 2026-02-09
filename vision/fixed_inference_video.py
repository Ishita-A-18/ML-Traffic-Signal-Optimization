import cv2
from pathlib import Path
from detector import VehicleDetector
from lane_mapper import assign_to_lanes

VIDEO_PATH = Path(__file__).parent / "test_video.mp4"

FIXED_GREEN = 30  # seconds

detector = VehicleDetector()
cap = cv2.VideoCapture(str(VIDEO_PATH))

assert cap.isOpened(), "Video not found"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vehicles = detector.detect(frame)
    lane_counts = assign_to_lanes(vehicles)

    y = 30
    for lane, count in lane_counts.items():
        cv2.putText(frame, f"{lane}: {count}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)
        y += 25

    cv2.putText(frame, f"FIXED Green Time: {FIXED_GREEN}s",
                (10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,255), 2)

    cv2.imshow("Fixed-Time Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
