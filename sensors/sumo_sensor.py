import traci
from sensors.base import LaneSensor

class SumoLaneSensor(LaneSensor):
    def __init__(self, lane_id):
        self.lane_id = lane_id

    def get_metrics(self):
        return {
            "queue": traci.lane.getLastStepHaltingNumber(self.lane_id),
            "waiting": traci.lane.getWaitingTime(self.lane_id),
            "speed": traci.lane.getLastStepMeanSpeed(self.lane_id)
        }
