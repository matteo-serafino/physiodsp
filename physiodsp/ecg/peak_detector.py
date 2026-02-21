from pandas import DataFrame
from numpy import (
    abs,
    convolve,
    diff,
    mean,
    round_
)
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks
)

from physiodsp.sensors.ecg import EcgData
from physiodsp.base import BaseAlgorithm
from physiodsp.dsp.convolution import mov_mean


class EcgPeakDetector(BaseAlgorithm):

    _algorithm_name = "EcgPeakDetector"
    _version = "v0.1.0"

    FILTER_ORDER: int = 3
    LOWER_F_CUT: int = 5
    UPPER_F_CUT: int = 15

    WIN_LEN_SEC: float = 0.150

    def __init__(self, settings=None) -> None:
        self.settings = settings
        return None

    def _normalize_signal(x):
        x_norm = x / max(abs(x))
        return x_norm

    def run(self, ecg: EcgData,):

        # Band pass filter settings
        [b, a] = butter(self.FILTER_ORDER, [self.LOWER_F_CUT/(ecg.fs * 0.5), self.UPPER_F_CUT/(ecg.fs * 0.5)], btype='bandpass')

        # Derivative filter settings
        impulse_response_deriv_filt = [-1/8, -2/8, 0, 2/8, 1/8]

        # Step 1a: Band pass filtering
        ecg_filt = filtfilt(b, a, ecg.values, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
        # Step 1b: Result Normalization
        ecg_filt = ecg_filt / max(abs(ecg_filt))

        # Step 2a: Derivative filtering
        ecg_deriv = convolve(ecg_filt, impulse_response_deriv_filt, mode='same')
        # Step 2b: Result Normalization
        ecg_deriv = ecg_deriv / max(abs(ecg_deriv))

        # Step 3: Signal squaring
        ecg_squared = ecg_deriv**2

        # Step 4a: Moving average filtering
        ecg_integrated = mov_mean(ecg_squared, int(self.WIN_LEN_SEC * self.ecg.fs), mode='same')
        # Step 4b: Result Normalization
        ecg_integrated = ecg_integrated / max(ecg_integrated)

        # Step 5: Peaks detection
        locs, _ = find_peaks(ecg_integrated, height=1 * mean(ecg_integrated), distance=round(0.2 * self.ecg.fs))

        # is_pvc, pvc_features = pvc_detector(ecg['value'], locs, fs=self.ecg.fs)

        rr = {
            "timestamps": ecg.timestamps[locs[1:]],
            "inter_beat_interval": round_(1000 * diff(locs) / ecg.fs)
        }
        rr['heart_rate'] = round_(60000 / rr['inter_beat_interval'])

        self.biomarker = DataFrame(rr)

        return self
