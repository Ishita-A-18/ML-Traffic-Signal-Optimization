import traci
from collections import defaultdict

class MetricsLogger:
    def __init__(self):
        self.waiting_times = defaultdict(float)
        self.depart_times = {}
        self.arrival_times = {}
        self.passed = 0

    def update(self):
        for veh in traci.vehicle.getIDList():
            self.waiting_times[veh] += traci.vehicle.getWaitingTime(veh)

        for veh in traci.simulation.getDepartedIDList():
            self.depart_times[veh] = traci.simulation.getTime()

        for veh in traci.simulation.getArrivedIDList():
            self.arrival_times[veh] = traci.simulation.getTime()
            self.passed += 1

    def results(self):
        avg_wait = sum(self.waiting_times.values()) / max(1, len(self.waiting_times))
        travel_times = [
            self.arrival_times[v] - self.depart_times[v]
            for v in self.arrival_times if v in self.depart_times
        ]
        avg_travel = sum(travel_times) / max(1, len(travel_times))
        return avg_wait, avg_travel, self.passed
