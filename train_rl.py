import traci, sumolib
from rl.agent import DQNAgent
from rl.env_utils import get_controlled_lanes, get_state, compute_reward

sumoBinary = sumolib.checkBinary("sumo")
traci.start([sumoBinary, "-c", "simulation/sim.sumocfg"])

lanes = get_controlled_lanes()
state_size = len(lanes) * 3
action_space = [10, 20, 30, 40, 50, 60]

agent = DQNAgent(state_size, len(action_space))

EPISODES = 50
MAX_STEPS = 2000

# Get number of phases once
tls = traci.trafficlight.getIDList()[0]
logic = traci.trafficlight.getAllProgramLogics(tls)[0]
num_phases = len(logic.phases)

for ep in range(EPISODES):
    traci.load(["-c", "simulation/sim.sumocfg"])
    
    # Initialize metrics with actual starting state
    total_q = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    total_w = sum(traci.lane.getWaitingTime(l) for l in lanes)
    prev_metrics = {"q": total_q, "w": total_w}
    
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
            sim_step += 1
        
        reward, prev_metrics = compute_reward(lanes, prev_metrics)
        next_state = get_state(lanes)
        
        agent.remember(state, action, reward, next_state)
        agent.replay()

    print(f"Episode {ep} | epsilon={agent.epsilon:.3f}")

agent.save("models/dqn_traffic.pt")
traci.close()