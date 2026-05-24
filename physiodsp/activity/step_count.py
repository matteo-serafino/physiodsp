from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.base import IMUData


class StepCount(BaseAlgorithm):
    """Step Count Algorithm (Not yet implemented)"""
    _algorithm_name = "StepCount"
    _version = "0.1.0"

    def run(self, data: IMUData):
        raise NotImplementedError("StepCount is not yet implemented")
