from numpy import abs
from pandas import DataFrame

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.base import IMUData


class PIMAlgorithm(BaseAlgorithm):
    """Proportional Integration Mode"""

    _algorithm_name = "PIMAlgorithm"
    _version = 'v0.1.0'
    _aggregation_window = 5

    def __init__(self) -> None:

        return None

    def estimate(self, data: IMUData):

        self.values_x = abs(data.x)
        self.values_y = abs(data.y)
        self.values_z = abs(data.z)

        return self

    def aggregate(self,
                  method: str = 'sum'
                  ):

        df = DataFrame(
            list(zip(self.data.timestamps, self.values_x, self.values_y, self.values_z)),
            columns=['timestamps', 'x', 'y', 'z']
        )

        df['timestamps'] = df[
            'timestamps'].apply(lambda x: x // self.aggregation_window)

        df_agg = df.groupby('timestamps')[["x", "y", "z"]].agg(method).reset_index(drop=False)

        self.biomarker_agg = df_agg

        return self
