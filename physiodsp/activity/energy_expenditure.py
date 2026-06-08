from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.base import IMUData


class EnergyExpenditure(BaseAlgorithm):
    """Energy Expenditure Algorithm (Not yet implemented)"""

    _algorithm_name = "EnergyExpenditure"
    _version = "0.1.0"

    def run(self, data: IMUData):
        raise NotImplementedError("EnergyExpenditure is not yet implemented")
