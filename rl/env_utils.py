import traci
import numpy as np

def get_controlled_lanes():
    tls = traci.trafficlight.getIDList()[0]
    return list(set(traci.trafficlight.getControlledLanes(tls)))

def get_state(lanes):
    state = []
    for l in lanes:
        q = traci.lane.getLastStepHaltingNumber(l)
        w = traci.lane.getWaitingTime(l)
        f = traci.lane.getLastStepVehicleNumber(l)
        state.extend([q, w, f])
    return np.array(state, dtype=np.float32)

def compute_reward(lanes, prev_metrics):
    total_q = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    total_w = sum(traci.lane.getWaitingTime(l) for l in lanes)
    throughput = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)

    dq = prev_metrics["q"] - total_q
    dw = prev_metrics["w"] - total_w

    reward = (
        0.4 * dw +
        0.4 * dq +
        0.2 * throughput
    )

    return reward, {"q": total_q, "w": total_w}
