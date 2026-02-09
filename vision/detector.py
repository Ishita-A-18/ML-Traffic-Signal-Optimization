from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

        # COCO vehicle class IDs
        self.vehicle_classes = [2, 3, 5, 7]  
        # car, motorcycle, bus, truck

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2))

        return detections
