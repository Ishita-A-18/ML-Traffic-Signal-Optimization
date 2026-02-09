import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np

from detector import VehicleDetector
from lane_mapper import assign_to_lanes
from rl.agent import DQNAgent

from pathlib import Path

# ----------------------------
# Load trained RL model
# ----------------------------
LANE_COUNT = 4
FEATURES_PER_LANE = 3   # vehicle count
STATE_SIZE = LANE_COUNT * FEATURES_PER_LANE
ACTIONS = [10, 20, 30, 40, 50, 60]

agent = DQNAgent(state_size=STATE_SIZE, action_size=len(ACTIONS))
agent.load("models/dqn_traffic.pt")
agent.epsilon = 0.0   # VERY IMPORTANT → no exploration

# ----------------------------
# Vision setup
# ----------------------------
detector = VehicleDetector()


VIDEO_PATH = Path(__file__).parent / "test_video.mp4"
cap = cv2.VideoCapture(str(VIDEO_PATH))

print("Video path:", VIDEO_PATH)
print("Video opened:", cap.isOpened())


frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detect vehicles
    vehicles = detector.detect(frame)

    # 2. Map to lanes
    lane_counts = assign_to_lanes(vehicles)

    # 3. Build RL state vector
    # Since vision only gives counts, replicate them for [queue, wait, flow]
    state = []
    for lane_id in sorted(lane_counts.keys()):
        count = lane_counts[lane_id]
        state.extend([count, count * 10, count])  # Approximate: queue, wait, flow
    state = np.array(state, dtype=np.float32)

    # 4. RL decision
    action_idx = agent.act(state)
    green_time = ACTIONS[action_idx]

    # ----------------------------
    # Visualization
    # ----------------------------
    y = 30
    for lane, count in lane_counts.items():
        cv2.putText(frame, f"{lane}: {count}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)
        y += 25

    cv2.putText(frame, f"RL Green Time: {green_time}s",
                (10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,255), 2)

    cv2.imshow("YOLO + RL Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
