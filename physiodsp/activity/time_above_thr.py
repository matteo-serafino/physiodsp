from numpy import abs, concatenate
from pandas import DataFrame
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.base import IMUData


class TimeAboveThrSettings(BaseModel):

    window_len: PositiveInt = Field(default=1, description="processing window length in seconds")

    aggregation_window: PositiveInt = Field(default=60, description="aggregation window length in seconds")

    threshold: PositiveFloat = Field(default=0.1, description="threshold in g")


class TimeAboveThr(BaseAlgorithm):
    """Time Above Threshold Algorithm"""

    _algorithm_name = "TimeAboveThrAlgorithm"
    _version = "v0.1.0"

    def __init__(self,
                 settings: TimeAboveThrSettings = TimeAboveThrSettings()
                 ) -> None:
        self.settings = settings
        self._window_len = settings.window_len
        self._aggregation_window = settings.aggregation_window
        return None

    def run(self, data: IMUData):

        imu_matrix = data.to_matrix()
        above_thr = (abs(imu_matrix) >= self.settings.threshold).astype(int)
        # Add timestamp column
        above_thr = concatenate([data.timestamps.reshape(-1, 1), above_thr], axis=1)

        self.values = DataFrame({"timestamps": above_thr[:, 0], "x": above_thr[:, 1], "y": above_thr[:, 2], "z": above_thr[:, 3]}).rolling(
            window=int(self._window_len * data.fs),
            step=int(self._window_len * data.fs),
            min_periods=int(self._window_len * data.fs),
            closed="left"
        ).agg({"timestamps": "max", "x": "sum", "y": "sum", "z": "sum"})[1:]

        return self

    def aggregate(self,
                  method: str = 'sum'
                  ):

        df = DataFrame(
            list(zip(self.data.timestamps, self.values_x, self.values_y, self.values_z)),
            columns=['timestamps', 'x', 'y', 'z']
        )

        df['timestamp'] = df[
            'timestamps'].apply(lambda x: x // self._aggregation_window)

        df_agg = df.groupby('timestamp')[["x", "y", "z"]].agg(method).reset_index(drop=False)

        self.biomarker_agg = df_agg

        return self
