import sys
import time
import subprocess
import pickle
import os
from pathlib import Path
from collections import defaultdict

log = defaultdict(list)

sys.path.append(str(Path(__file__).parent.parent))

import traci
import cv2
import numpy as np

from rl.agent import DQNAgent
from rl.reward import compute_reward
from vision.lane_density import get_lane_density

# ============= LANE MAPPING =============
# Define which lane is monitored by video and which by SUMO
VIDEO_LANE = "north_in"
SUMO_LANES = []  # Will auto-detect from SUMO

# ============= CONFIG =============
SUMO_BINARY = "sumo-gui"
SUMO_CFG = "simulation/sim.sumocfg"
PORT = 51823

MODEL_PATH = "models/dqn_traffic.pt"
ACTIONS = [10, 20, 30, 40, 50, 60]
MAX_STEPS = 2000
VIDEO_PATH = "test_video.mp4"

# ====================================

def start_sumo():
    subprocess.Popen([
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "--step-length", "1",
        "--remote-port", str(PORT)
    ])
    time.sleep(2)

def get_hybrid_state(video_density, sumo_lanes):
    """
    Combine video-based and SUMO-based state into single vector.
    
    Args:
        video_density: dict {"north_in": vehicle_count}
        sumo_lanes: list of lane IDs from SUMO
    
    Returns:
        np.array of shape (12,) for trained model
        [video_features(3), sumo_lane1(3), sumo_lane2(3), sumo_lane3(3)]
    """
    state = []
    
    # Video-monitored lane (3 features)
    video_count = video_density.get(VIDEO_LANE, 0)
    state.append(float(video_count))                    # queue
    state.append(float(video_count * 2.0))              # proxy waiting time
    state.append(max(1.0, 15.0 - float(video_count)))   # proxy speed (inverse)
    
    # SUMO-monitored lanes (3 features each)
    for lane_id in sumo_lanes:
        queue = traci.lane.getLastStepHaltingNumber(lane_id)
        wait = traci.lane.getWaitingTime(lane_id)
        speed = traci.lane.getLastStepMeanSpeed(lane_id)
        
        state.append(float(queue))
        state.append(float(wait))
        state.append(float(speed))
    
    return np.array(state, dtype=np.float32)

def main():
    global SUMO_LANES
    
    start_sumo()
    traci.init(PORT)

    tls_id = traci.trafficlight.getIDList()[0]
    phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)

    print(f"Connected to SUMO | TLS={tls_id} | phases={phases}")

    # Get unique controlled lanes
    all_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    if len(all_lanes) == 0:
        print("ERROR: No controlled lanes found")
        traci.close()
        return

    # Separate into VIDEO lane + SUMO lanes
    # For now: use first 3 real lanes as SUMO lanes, video is synthetic
    SUMO_LANES = all_lanes[:3] if len(all_lanes) >= 3 else all_lanes
    
    print(f"Video lane: {VIDEO_LANE}")
    print(f"SUMO lanes: {SUMO_LANES}")
    print(f"State size: 4 lanes × 3 features = 12")
    
    # Load trained RL agent
    state_size = 12  # 4 lanes × 3 features
    agent = DQNAgent(state_size, len(ACTIONS))
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0
    
    print(f"✅ Loaded RL model from {MODEL_PATH}\n")
    
    # Initialize video capture
    video_cap = cv2.VideoCapture(VIDEO_PATH)
    
    sim_step = 0
    decision = 0
    total_reward = 0
    
    print("Starting hybrid RL control (video + SUMO)...\n")
    
    while sim_step < MAX_STEPS:
        # Get video density
        video_density = get_lane_density(cap=video_cap)
        
        # Build hybrid state vector
        state = get_hybrid_state(video_density, SUMO_LANES)
        
        # RL decision
        action_idx = agent.act(state)
        green_time = ACTIONS[action_idx]
        
        # Change phase
        current_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = (current_phase + 1) % phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        
        # Hold green for decided duration
        for _ in range(green_time):
            if sim_step >= MAX_STEPS:
                break
            traci.simulationStep()
            sim_step += 1
        
        # Compute reward (negative waiting time)
        total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in SUMO_LANES)
        total_wait = sum(traci.lane.getWaitingTime(l) for l in SUMO_LANES)
        reward = -(0.7 * total_queue + 0.3 * total_wait)
        total_reward += reward
        decision += 1

        # Get video count for logging
        video_count = video_density.get(VIDEO_LANE, 0)

        log["step"].append(sim_step)
        log["green"].append(green_time)
        log["phase"].append(next_phase)
        log["queue"].append(
            sum(traci.lane.getLastStepHaltingNumber(l) for l in SUMO_LANES)
        )
        log["reward"].append(reward)
        log["video_count"].append(video_count)
        
        # Log every 10 decisions
        if decision % 10 == 0:
            avg_reward = total_reward / decision
            print(
                f"Decision {decision:3d} | Step {sim_step:4d} | "
                f"Green={green_time}s | Video_count={video_count:2d} | "
                f"Queue={total_queue:2d} | Avg_Reward={avg_reward:.2f}"
            )
    
    video_cap.release()
    traci.close()
    
    print(f"\n✅ Simulation complete!")
    print(f"Total decisions: {decision}")
    print(f"Final average reward: {total_reward / decision:.2f}")


if __name__ == "__main__":
    main()
os.makedirs("logs", exist_ok=True)
with open("logs/run_log.pkl", "wb") as f:
    pickle.dump(log, f)
print("📁 Logs saved to logs/run_log.pkl")