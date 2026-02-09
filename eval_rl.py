import traci, sumolib
import numpy as np
from rl.agent import DQNAgent
from rl.env_utils import get_controlled_lanes, get_state

sumoBinary = sumolib.checkBinary("sumo")
traci.start([sumoBinary, "-c", "simulation/sim.sumocfg"])

lanes = get_controlled_lanes()
state_size = len(lanes) * 3
action_space = [10, 20, 30, 40, 50, 60]

# Get number of phases
tls = traci.trafficlight.getIDList()[0]
logic = traci.trafficlight.getAllProgramLogics(tls)[0]
num_phases = len(logic.phases)

agent = DQNAgent(state_size, len(action_space))
agent.load("models/dqn_traffic.pt")
agent.epsilon = 0.0  # No exploration during evaluation

total_wait = 0
throughput = 0
MAX_STEPS = 2000
sim_step = 0

while sim_step < MAX_STEPS:
    state = get_state(lanes)
    
    action = agent.act(state)
    green = action_space[action]
    
    # Change to next phase
    current_phase = traci.trafficlight.getPhase(tls)
    traci.trafficlight.setPhase(tls, (current_phase + 1) % num_phases)
    
    # Hold phase for green duration
    for _ in range(green):
        if sim_step >= MAX_STEPS:
            break
        traci.simulationStep()
        
        # Collect metrics at each step
        total_wait += sum(traci.lane.getWaitingTime(l) for l in lanes)
        throughput += sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
        
        sim_step += 1

print("RL RESULTS")
print(f"Avg waiting time: {total_wait / MAX_STEPS:.3f}")
print(f"Throughput: {throughput}")

traci.close()