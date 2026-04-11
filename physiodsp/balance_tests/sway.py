import numpy as np
from pandas import DataFrame
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat
from scipy.signal import butter, sosfiltfilt

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.accelerometer import AccelerometerData

CHI_SQUARED = 4.605
ROUND_DIGITS = 6


class SwaySettings(BaseModel):

    filter_order: PositiveInt = Field(default=4, description="Butterworth filter order for low-pass filtering")

    filter_high_freq: PositiveFloat = Field(default=2.5, description="Cutoff frequency in Hz")


class Sway(BaseAlgorithm):

    _algorithm_name = "Sway"
    _version = "v0.1.0"

    def __init__(self, settings: SwaySettings = SwaySettings()) -> None:
        self.settings = settings
        return None

    def run(self, accelerometer: AccelerometerData, sensor_height: float):
        """Run the sway analysis on accelerometer data recorded during a balance test.

        Medio-lateral (ML) displacement is estimated from the X axis of the accelerometer,
        while antero-posterior (AP) displacement is estimated from the Z axis. Both
        displacements are approximated by multiplying the respective axis signal by the
        sensor height (small-angle approximation). The signals are then low-pass filtered,
        mean-centred, and used to extract stabilometric indices and a 95% confidence ellipse.

        Args:
            accelerometer (AccelerometerData): Triaxial accelerometer data. The X axis must
                be aligned with the medio-lateral direction and the Z axis with the
                antero-posterior direction.
            sensor_height (float): Height of the sensor above the ground in meters, used to
                convert acceleration to linear displacement.

        Returns:
            Sway: The instance itself with the `biomarker` attribute set to a DataFrame
                containing stabilometric indices (average distance, RMS distance, total
                distance, average velocity) for the full, ML, and AP paths, joined with
                the 95% confidence ellipse metrics (area, angle, semi-major and semi-minor
                axes), and metadata columns (timestamp_start, timestamp_end, duration_s).
        """

        # Medio-lateral and Antero-posterior displacements in meters
        ml_disp = np.dot(accelerometer.x, sensor_height)
        ap_disp = np.dot(accelerometer.z, sensor_height)

        sos = butter(self.settings.filter_order,
                     self.settings.filter_high_freq,
                     btype='low',
                     fs=accelerometer.fs,
                     output='sos')

        ml_disp_filt = sosfiltfilt(sos, ml_disp)
        ap_disp_filt = sosfiltfilt(sos, ap_disp)

        ml_offset = np.round(np.mean(ml_disp_filt), ROUND_DIGITS)
        ap_offset = np.round(np.mean(ap_disp_filt), ROUND_DIGITS)

        ml_disp_no_mean = ml_disp_filt - ml_offset

        ap_disp_no_mean = ap_disp_filt - ap_offset

        metrics_df = self.__index_extraction(ml_disp_no_mean, ap_disp_no_mean, accelerometer.fs)

        ellipse_metrics_df = self.__get_ellipse_area(ml_disp_no_mean, ap_disp_no_mean)

        metrics_df = metrics_df.join(ellipse_metrics_df)

        metrics_df.insert(1, "timestamp_start", accelerometer.timestamps[0])
        metrics_df.insert(2, "timestamp_end", accelerometer.timestamps[-1])
        metrics_df.insert(3, "duration_s", (accelerometer.timestamps[-1] - accelerometer.timestamps[0]))

        self.biomarker = metrics_df.copy()

        return self

    def __index_extraction(self, ml: np.ndarray, ap: np.ndarray, fs: int):
        """Extract sway metrics from medio-lateral and antero-posterior displacement signals.

        Args:
            ml (np.ndarray): Medio-lateral displacement signal in meters.
            ap (np.ndarray): Antero-posterior displacement signal in meters.
            fs (int): Sampling frequency in Hz.

        Returns:
            DataFrame: Table of sway indices (average distance, RMS distance, total distance,
                average velocity) for the full path, ML path, and AP path.
        """

        metrics = {
            "index": ["full_path", "ml_path", "ap_path"],
            "average_distance_m": [],
            "rms_distance_m": [],
            "total_distance_m": [],
            "average_velocity_m_s": []
        }

        rd = np.sqrt(ml**2 + ap**2)

        metrics["average_distance_m"].append(np.round(np.mean(rd), ROUND_DIGITS))
        metrics["average_distance_m"].append(np.round(np.mean(np.abs(ml)), ROUND_DIGITS))
        metrics["average_distance_m"].append(np.round(np.mean(np.abs(ap)), ROUND_DIGITS))

        metrics["rms_distance_m"].append(np.round(np.sqrt(np.sum(rd**2) / len(ml)), ROUND_DIGITS))
        metrics["rms_distance_m"].append(np.round(np.sqrt(np.sum(ml**2) / len(ml)), ROUND_DIGITS))
        metrics["rms_distance_m"].append(np.round(np.sqrt(np.sum(ap**2) / len(ap)), ROUND_DIGITS))

        metrics["total_distance_m"].append(np.round(np.sum(np.sqrt((np.diff(ml))**2 + (np.diff(ap))**2)), ROUND_DIGITS))
        metrics["total_distance_m"].append(np.round(np.sum(np.abs(np.diff(ml))), ROUND_DIGITS))
        metrics["total_distance_m"].append(np.round(np.sum(np.abs(np.diff(ap))), ROUND_DIGITS))

        metrics["average_velocity_m_s"].append(np.round(metrics["total_distance_m"][0] / (len(ml) / fs), ROUND_DIGITS))
        metrics["average_velocity_m_s"].append(np.round(metrics["total_distance_m"][1] / (len(ml) / fs), ROUND_DIGITS))
        metrics["average_velocity_m_s"].append(np.round(metrics["total_distance_m"][2] / (len(ap) / fs), ROUND_DIGITS))

        return DataFrame(metrics)

    def __get_ellipse_area(self, ml: np.ndarray, ap: np.ndarray):
        """Calculate the area of the ellipse representing the sway.

        Args:
            ml (np.ndarray): Medio-lateral displacement signal in meters.
            ap (np.ndarray): Antero-posterior displacement signal in meters.

        Returns:
            DataFrame: Table of ellipse metrics (area, angle, semi-major axis, semi-minor axis).
        """

        m = np.vstack((ml, ap))

        sigma = np.cov(m)

        U, S, V = np.linalg.svd(sigma)

        ellipse_area = np.round(np.pi * CHI_SQUARED * np.sqrt(S[0] * S[1]), ROUND_DIGITS)
        theta = np.round(np.arctan(U[1][0] / U[0][0]), ROUND_DIGITS)
        a = np.round(np.sqrt(S[0] * CHI_SQUARED), ROUND_DIGITS)
        b = np.round(np.sqrt(S[1] * CHI_SQUARED), ROUND_DIGITS)

        metrics = {
            "ellipse_area_m2": [ellipse_area],
            "theta_rad": [theta],
            "a_m": [a],
            "b_m": [b]
        }

        return DataFrame(metrics)
