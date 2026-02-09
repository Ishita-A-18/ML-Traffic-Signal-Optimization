import cv2
from vision.detector import VehicleDetector


def get_lane_density(video_path=None, cap=None):
    """
    Extract vehicle count (density) from video frame.
    
    Returns:
        dict: {"north_in": count}
    """
    if cap is None:
        cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    
    detector = VehicleDetector()
    detections = detector.detect(frame)
    
    # All vehicles detected are in the "north_in" lane (single camera)
    vehicle_count = len(detections)
    
    return {"north_in": vehicle_count}
