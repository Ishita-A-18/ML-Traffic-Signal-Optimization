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

from vision.lane_density import get_lane_density

# ============= LANE MAPPING =============
VIDEO_LANE = "north_in"
SUMO_LANES = []  # Will auto-detect from SUMO

# ============= CONFIG =============
SUMO_BINARY = "sumo-gui"
SUMO_CFG = "simulation/sim.sumocfg"
PORT = 51824  # Different port from hybrid

# Fixed timing (baseline)
FIXED_GREEN_DURATION = 30  # seconds per phase
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

    # Use first 3 lanes as SUMO lanes
    SUMO_LANES = all_lanes[:3] if len(all_lanes) >= 3 else all_lanes
    
    print(f"Video lane: {VIDEO_LANE}")
    print(f"SUMO lanes: {SUMO_LANES}")
    print(f"Fixed green duration: {FIXED_GREEN_DURATION}s\n")
    
    # Initialize video capture
    video_cap = cv2.VideoCapture(VIDEO_PATH)
    
    sim_step = 0
    decision = 0
    total_reward = 0
    green_counter = 0  # Counter for fixed green duration
    
    print("Starting FIXED (baseline) control...\n")
    
    while sim_step < MAX_STEPS:
        # Get video density (for logging, but not used for control)
        from vision.lane_density import get_lane_density
        video_density = get_lane_density(cap=video_cap)
        video_count = video_density.get(VIDEO_LANE, 0)
        
        # Fixed timing: change phase every FIXED_GREEN_DURATION steps
        if green_counter == 0:
            current_phase = traci.trafficlight.getPhase(tls_id)
            next_phase = (current_phase + 1) % phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            decision += 1
            green_counter = FIXED_GREEN_DURATION
            current_decision_phase = next_phase
        
        # Run one SUMO step
        traci.simulationStep()
        sim_step += 1
        green_counter -= 1
        
        # Compute metrics
        total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in SUMO_LANES)
        total_wait = sum(traci.lane.getWaitingTime(l) for l in SUMO_LANES)
        reward = -(0.7 * total_queue + 0.3 * total_wait)
        total_reward += reward
        
        # Log data
        log["step"].append(sim_step)
        log["green"].append(FIXED_GREEN_DURATION)  # Always same
        log["phase"].append(traci.trafficlight.getPhase(tls_id))
        log["queue"].append(total_queue)
        log["reward"].append(reward)
        log["video_count"].append(video_count)
        
        # Print every 10 decisions
        if decision % 10 == 0:
            avg_reward = total_reward / decision
            print(
                f"Decision {decision:3d} | Step {sim_step:4d} | "
                f"Green={FIXED_GREEN_DURATION}s | Video_count={video_count:2d} | "
                f"Queue={total_queue:2d} | Avg_Reward={avg_reward:.2f}"
            )
    
    video_cap.release()
    traci.close()
    
    print(f"\n✅ Fixed control simulation complete!")
    print(f"Total decisions: {decision}")
    print(f"Final average reward: {total_reward / decision:.2f}")
    print(f"Total queue time: {sum(log['queue'])}")

    # Save logs
    os.makedirs("logs", exist_ok=True)
    with open("logs/fixed_control_log.pkl", "wb") as f:
        pickle.dump(log, f)
    print(f"📊 Logs saved to logs/fixed_control_log.pkl")


if __name__ == "__main__":
    main()
