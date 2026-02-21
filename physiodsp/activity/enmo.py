from pandas import Series, DataFrame
from pydantic import BaseModel, Field, PositiveInt

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.accelerometer import AccelerometerData


class ENMOSettings(BaseModel):

    window_len: PositiveInt = Field(default=1, description="processing window length in seconds")

    aggregation_window: PositiveInt = Field(default=60, description="aggregation window length in seconds")


class ENMO(BaseAlgorithm):
    """Euclidean Norm Minus One"""

    _algorithm_name = "ENMO"
    _version = "v0.1.0"

    def __init__(self,
                 settings: ENMOSettings = ENMOSettings()
                 ) -> None:
        self.settings = settings
        self._window_len = settings.window_len
        self._aggregation_window = settings.aggregation_window
        return None

    def run(self, accelerometer: AccelerometerData):
        """_summary_

        Args:
            accelerometer (AccelerometerData): _description_

        Returns:
            ENMO: _description_
        """

        enmo = accelerometer.magnitude - 1
        enmo[enmo < 0] = 0

        self.timestamps = Series(accelerometer.timestamps).rolling(
            window=int(self._window_len * accelerometer.fs),
            step=int(self._window_len * accelerometer.fs),
            min_periods=int(self._window_len * accelerometer.fs),
            closed="left"
        ).max()[1:]

        self.values = Series(enmo).rolling(
            window=int(self._window_len * accelerometer.fs),
            step=int(self._window_len * accelerometer.fs),
            min_periods=int(self._window_len * accelerometer.fs),
            closed="left"
        ).mean()[1:]

        self.biomarker = DataFrame(
            list(zip(self.timestamps, self.values)),
            columns=['timestamps', 'values']
        )

        return self

    def aggregate(self,
                  method: str = 'mean'
                  ):
        super().aggregate(
            self.timestamps,
            self.values,
            method
        )
        return self
