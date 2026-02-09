import traci
import sumolib
from metrics import MetricsLogger

sumoBinary = sumolib.checkBinary("sumo")

traci.start([
    sumoBinary,
    "-c", "simulation/sim.sumocfg",
    "--start"
])

logger = MetricsLogger()

step = 0
current_green_end = 0
FIXED_GREEN = 30  # baseline

while step < 2000:
    traci.simulationStep()
    logger.update()

    tls_id = traci.trafficlight.getIDList()[0]

    if step >= current_green_end:
        phase = traci.trafficlight.getPhase(tls_id)
        traci.trafficlight.setPhase(tls_id, (phase + 1) % 4)
        current_green_end = step + FIXED_GREEN

    step += 1

print("FIXED RESULTS:", logger.results())
traci.close()
