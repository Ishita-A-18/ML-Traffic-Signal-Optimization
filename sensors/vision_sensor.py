import cv2
import numpy as np
from sensors.base import LaneSensor
from vision.vehicle_detector import VehicleDetector
from vision.lane_mapper import assign_to_lanes

class VisionLaneSensor(LaneSensor):
    def __init__(self, video_path, lane_id=0):
        self.cap = cv2.VideoCapture(video_path)
        self.detector = VehicleDetector()
        self.lane_id = lane_id

    def get_metrics(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        vehicles = self.detector.detect(frame)
        lane_counts = assign_to_lanes(vehicles)

        count = lane_counts.get(self.lane_id, 0)

        # Approximate metrics from vision
        return {
            "queue": count,
            "waiting": count * 10.0,
            "speed": max(1.0, 15.0 - count)
        }
