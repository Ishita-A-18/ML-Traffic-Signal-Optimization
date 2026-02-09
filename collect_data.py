import traci
import sumolib
import numpy as np
import pandas as pd
import os

sumoBinary = sumolib.checkBinary("sumo")

traci.start([
    sumoBinary,
    "-c", "simulation/sim.sumocfg",
    "--start"
])

data = []

step = 0
current_green_end = 0

while step < 2000:
    traci.simulationStep()

    tls_id = traci.trafficlight.getIDList()[0]
    lanes = traci.trafficlight.getControlledLanes(tls_id)

    lane_states = []
    for lane in lanes:
        queue = traci.lane.getLastStepHaltingNumber(lane)
        waiting = traci.lane.getWaitingTime(lane)
        flow = traci.lane.getLastStepVehicleNumber(lane)
        lane_states.append([queue, waiting, flow])

    state = np.array(lane_states).flatten()

    total_queue = sum(state[::3])
    green_time = 10 if total_queue < 5 else 15

    if step >= current_green_end:
        data.append(list(state) + [green_time])
        current_green_end = step + green_time

    step += 1

traci.close()

cols = [f"f{i}" for i in range(len(state))] + ["green_time"]
df = pd.DataFrame(data, columns=cols)

os.makedirs("data", exist_ok=True)
df.to_csv("data/dataset.csv", index=False)

print("Saved dataset with", len(df), "samples")
