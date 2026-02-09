import cv2
from detector import VehicleDetector
from lane_mapper import assign_to_lanes

detector = VehicleDetector()
cap = cv2.VideoCapture("test_video.mp4")

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    vehicles = detector.detect(frame)
    lane_counts = assign_to_lanes(vehicles)

    print(f"Frame {frame_id} | {lane_counts}")

    # Optional visualization
    for (x1, y1, x2, y2) in vehicles:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("YOLO + Lane Mapping", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
