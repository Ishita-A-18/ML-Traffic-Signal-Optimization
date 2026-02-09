import numpy as np
import traci

def get_state(lanes):
    state = []
    for l in lanes:
        queue = traci.lane.getLastStepHaltingNumber(l)
        waiting = traci.lane.getWaitingTime(l)
        flow = traci.lane.getLastStepVehicleNumber(l)
        state.extend([queue, waiting, flow])
    return np.array(state, dtype=np.float32)

def compute_wait(lanes):
    return sum(traci.lane.getWaitingTime(l) for l in lanes)

def compute_reward(prev_wait, curr_wait, arrived, phase_changed):
    # normalize components
    wait_term = (prev_wait - curr_wait) / 100.0
    arrive_term = arrived / 10.0
    switch_penalty = -0.2 if phase_changed else 0.0

    reward = wait_term + arrive_term + switch_penalty
    return float(np.clip(reward, -1.0, 1.0))
