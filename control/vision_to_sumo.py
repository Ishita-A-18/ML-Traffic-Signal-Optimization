import traci
import subprocess
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rl.agent import DQNAgent
from rl.env_utils import get_controlled_lanes, get_state, compute_reward

SUMO_BINARY = "sumo-gui"  # Use GUI to visualize
SUMO_CFG = "simulation/sim.sumocfg"
MODEL_PATH = "models/dqn_traffic.pt"
ACTION_SPACE = [10, 20, 30, 40, 50, 60]

def start_sumo():
    sumo_cmd = [
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "--step-length", "1",
        "--remote-port", "51823"
    ]
    subprocess.Popen(sumo_cmd)
    time.sleep(2)

def main():
    start_sumo()

    print("Connecting to SUMO...")
    traci.init(51823)
    print("Connected!")
    
    tls_list = traci.trafficlight.getIDList()
    if len(tls_list) == 0:
        print("ERROR: No traffic lights!")
        traci.close()
        return
    
    tls_id = tls_list[0]
    print(f"Using traffic light: {tls_id}")
    
    # Get controlled lanes and phase info
    lanes = get_controlled_lanes()
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    num_phases = len(logic.phases)
    
    print(f"Controlled lanes: {len(lanes)}")
    print(f"Phases: {num_phases}")
    print(f"State size: {len(lanes) * 3}")
    
    # Load trained RL agent
    state_size = len(lanes) * 3
    agent = DQNAgent(state_size, len(ACTION_SPACE))
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0  # No exploration
    print(f"✅ Loaded RL model from {MODEL_PATH}\n")
    
    # Initialize metrics
    prev_metrics = {
        "q": sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes),
        "w": sum(traci.lane.getWaitingTime(l) for l in lanes)
    }
    
    sim_step = 0
    decision_count = 0
    total_reward = 0
    
    print("Starting RL control...\n")
    
    while sim_step < 2000:
        # Get current state
        state = get_state(lanes)
        
        # RL decision
        action_idx = agent.act(state)
        green_duration = ACTION_SPACE[action_idx]
        
        # Change phase
        current_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = (current_phase + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        
        # Hold green for decided duration
        for _ in range(green_duration):
            if sim_step >= 2000:
                break
            traci.simulationStep()
            sim_step += 1
        
        # Compute reward
        reward, prev_metrics = compute_reward(lanes, prev_metrics)
        total_reward += reward
        decision_count += 1
        
        # Log every 10 decisions
        if decision_count % 10 == 0:
            avg_reward = total_reward / decision_count
            queues = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            print(f"Decision {decision_count} | Step {sim_step} | "
                  f"Green={green_duration}s | Queue={queues} | "
                  f"Avg Reward={avg_reward:.1f}")
    
    print(f"\n✅ Simulation complete!")
    print(f"Total decisions: {decision_count}")
    print(f"Average reward: {total_reward / decision_count:.2f}")
    
    traci.close()

if __name__ == "__main__":
    main()