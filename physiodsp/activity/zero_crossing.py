from numpy import diff, abs
from pandas import DataFrame
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat
from scipy.signal import butter, sosfilt, sosfilt_zi

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.base import IMUData


class ZeroCrossingSettings(BaseModel):

    window_len: PositiveInt = Field(default=1, description="processing window length in seconds")

    aggregation_window: PositiveInt = Field(default=60, description="aggregation window length in seconds")

    zero_crossing_thr: PositiveFloat = Field(default=0.05, description="ero crossing threshold in g")

    filter_order: PositiveInt = Field(default=4, description="Butterworth filter order for bandpass filtering")

    filter_low_freq: PositiveFloat = Field(default=0.3, description="Lower cutoff frequency in Hz")

    filter_high_freq: PositiveFloat = Field(default=3.5, description="Upper cutoff frequency in Hz")


class ZeroCrossing(BaseAlgorithm):
    """Zero Crossing Algorithm"""

    _algorithm_name = "ZeroCrossingAlgorithm"
    _version = "v0.1.0"

    def __init__(self,
                 settings: ZeroCrossingSettings = ZeroCrossingSettings(),
                 ) -> None:

        self.settings = settings
        self._window_len = settings.window_len
        self._aggregation_window = settings.aggregation_window
        self.zero_crossing_thr = settings.zero_crossing_thr
        return None

    def __preprocess_imu(self, imu_matrix, fs: int):
        """
        Apply bandpass Butterworth filter to IMU data.

        Reference: 0.3-3.5 Hz pass band is commonly used for human movement analysis
        to isolate low-frequency body motion while removing drift and high-frequency noise.
        Args:
            imu_matrix: (N, 3) array with columns [x, y, z]
            fs: Sampling frequency of the IMU data in Hz

        Returns:
            Filtered (N, 3) array
        """
        # Design bandpass filter using second-order sections for numerical stability
        sos = butter(self.settings.filter_order,
                     [self.settings.filter_low_freq, self.settings.filter_high_freq],
                     btype='band',
                     fs=fs,
                     output='sos')

        # Initialize filter states for each axis
        zi = sosfilt_zi(sos)

        # Apply filter to each axis using sosfilt (not filtfilt)
        filtered_matrix = imu_matrix.copy()
        for i in range(imu_matrix.shape[1]):
            filtered_matrix[:, i], _ = sosfilt(sos, imu_matrix[:, i], zi=zi * imu_matrix[0, i])

        return filtered_matrix

    def run(self, data: IMUData):

        # Apply bandpass filter to IMU data
        imu_matrix = data.to_matrix()
        imu_matrix_filtered = self.__preprocess_imu(imu_matrix, fs=data.fs)
        # Compute zero crossings for each axis
        zcr = abs(diff((imu_matrix_filtered >= self.zero_crossing_thr).astype(int), axis=0))

        self.values = DataFrame({"timestamps": data.timestamps[1:], "x": zcr[:, 0], "y": zcr[:, 1], "z": zcr[:, 2]}).rolling(
            window=int(self._window_len * data.fs),
            step=int(self._window_len * data.fs),
            min_periods=int(self._window_len * data.fs),
            closed="left"
        ).agg({"timestamps": "max", "x": "sum", "y": "sum", "z": "sum"})[1:]

        self.biomarker = self.values.copy()

        return self

    def aggregate(self,
                  method: str = 'sum'
                  ):

        df = self.values.copy()

        df['timestamps'] = df[
            'timestamps'].apply(lambda x: (x // self._aggregation_window) * self._aggregation_window)

        df_agg = df.groupby('timestamps')[["x", "y", "z"]].agg(method).reset_index(drop=False)

        self.biomarker_agg = df_agg

        return self
