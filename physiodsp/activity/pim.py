from numpy import abs
from pandas import DataFrame

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.base import IMUData


class PIMAlgorithm(BaseAlgorithm):
    """Proportional Integration Mode"""

    _algorithm_name = "PIMAlgorithm"
    _version = "0.1.0"
    _aggregation_window = 5

    def run(self, data: IMUData):

        self.values_x = abs(data.x)
        self.values_y = abs(data.y)
        self.values_z = abs(data.z)

        return self

    def aggregate(self,
                  method: str = 'sum'
                  ):

        df = DataFrame({
            'timestamps': self.data.timestamps,
            'x': self.values_x,
            'y': self.values_y,
            'z': self.values_z
        })

        df['timestamps'] = df['timestamps'] // self.aggregation_window

        df_agg = df.groupby('timestamps')[["x", "y", "z"]].agg(method).reset_index(drop=False)

        self.biomarker_agg = df_agg

        return self
