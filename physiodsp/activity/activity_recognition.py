from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.base import IMUData


class ActivityRecognition(BaseAlgorithm):
    """Activity Recognition Algorithm (Not yet implemented)"""

    _algorithm_name = "ActivityRecognition"
    _version = "0.1.0"

    def run(self, data: IMUData):
        raise NotImplementedError("ActivityRecognition is not yet implemented")
