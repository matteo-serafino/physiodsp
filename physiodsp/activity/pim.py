from numpy import abs
from pandas import DataFrame
from pydantic import BaseModel

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.base import IMUData


class PIMSettings(BaseModel):
    """PIM Algorithm Settings"""
    aggregation_window: int = 5


class PIMAlgorithm(BaseAlgorithm):
    """Proportional Integration Mode"""

    _algorithm_name = "PIMAlgorithm"
    _version = "0.1.0"

    def __init__(self,
                 settings: PIMSettings = PIMSettings()
                 ) -> None:
        self.settings = settings
        self._aggregation_window = settings.aggregation_window
        return None

    def run(self, data: IMUData):
        self.data = data
        self.values_x = abs(data.x)
        self.values_y = abs(data.y)
        self.values_z = abs(data.z)

        # Raw biomarker: combine absolute values for raw signal representation
        self.biomarker = DataFrame({
            'timestamps': data.timestamps,
            'x': self.values_x,
            'y': self.values_y,
            'z': self.values_z
        })

        return self

    def aggregate(self,
                  method: str = 'sum'
                  ):

        df = self.biomarker.copy()

        df['timestamps'] = (df['timestamps'] // self.aggregation_window) * self.aggregation_window

        df_agg = df.groupby('timestamps')[["x", "y", "z"]].agg(method).reset_index(drop=False)

        self.biomarker_agg = df_agg

        return self
