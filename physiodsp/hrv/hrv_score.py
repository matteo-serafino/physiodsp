from typing import Literal
import numpy as np
from pandas import DataFrame
from pydantic import BaseModel, Field, PositiveInt
from scipy.stats import percentileofscore

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.hrv import HrvData


class HrvScoreSettings(BaseModel):

    window_len: PositiveInt = Field(default=30, description="processing window length in days")

    method: Literal["sigmoid", "percentile"] = Field(default="sigmoid", description="Method to map the HRV score in the 0-100 range")


class HrvScore(BaseAlgorithm):
    """HRV Score Algorithm"""

    _algorithm_name = "HrvScoreAlgorithm"
    _version = "v0.1.0"

    def __init__(self, settings: HrvScoreSettings = HrvScoreSettings()) -> None:
        self.settings = settings
        self._window_len = settings.window_len
        return None

    def run(self, data: HrvData):

        hrv_score = self._compute_hrv_score(data.values[-1], data.values[-self._window_len:-1])

        self.biomarker_agg = DataFrame({"timestamps": data.timestamps[-1], "hrv_score": hrv_score})

        return self

    def _compute_hrv_score(self, current_rmssd, last_30_days_rmssd):
        """
        Compute 0-100 HRV score based on 30-day baseline

        Args:
            current_rmssd: Today's RMSSD value (ms)
            last_30_days_rmssd: Array of RMSSD values from past 30 days

        Returns:
            hrv_score: Score from 0-100
        """
        # Step 1: Baseline and variability
        baseline = np.median(last_30_days_rmssd)
        std = np.std(last_30_days_rmssd)
        cv = std / baseline if baseline > 0 else 0

        # Step 2: Z-score
        z_score = (current_rmssd - baseline) / std if std > 0 else 0

        # Step 3: Map to 0-100 using sigmoid
        if self.settings.method in ["sigmoid"]:
            base_score = 50 + 30 * np.tanh(0.5 * z_score)
        else:
            # Alternative: percentile method
            base_score = percentileofscore(last_30_days_rmssd, current_rmssd)

        # Step 4: Trend bonus
        first_10_mean = np.mean(last_30_days_rmssd[:10])
        last_10_mean = np.mean(last_30_days_rmssd[-10:])
        trend = (last_10_mean - first_10_mean) / first_10_mean if first_10_mean > 0 else 0

        if trend > 0.10:
            trend_bonus = 10
        elif trend > 0.05:
            trend_bonus = 5
        elif trend < -0.05:
            trend_bonus = -5
        else:
            trend_bonus = 0

        # Step 5: Stability penalty
        if cv > 0.15:
            stability_penalty = -10
        elif cv > 0.10:
            stability_penalty = -5
        else:
            stability_penalty = 0

        # Final score
        hrv_score = base_score + trend_bonus + stability_penalty
        hrv_score = np.clip(hrv_score, 0, 100)

        return round(hrv_score)
