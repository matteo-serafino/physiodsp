from typing import Literal

import numpy as np
from pandas import DataFrame
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat

from physiodsp.base import BaseAlgorithm


class ActivityScoreSettings(BaseModel):
    """Configuration settings for Activity Score algorithm"""

    # Baseline window (number of days to use for personalization)
    baseline_window_days: PositiveInt = Field(default=30, description="Number of days to use for computing baselines")

    # Factor weights (must sum to 1.0)
    step_weight: PositiveFloat = Field(default=0.25, description="Weight for step count factor")
    sleep_weight: PositiveFloat = Field(default=0.35, description="Weight for sleep factor")
    training_weight: PositiveFloat = Field(default=0.25, description="Weight for training time factor")
    resting_weight: PositiveFloat = Field(default=0.15, description="Weight for resting time factor")

    # Step targets (personalized)
    baseline_daily_steps: PositiveInt = Field(default=8000, description="User's baseline daily steps")
    step_ceiling: PositiveInt = Field(default=15000, description="Maximum steps for optimal score")

    # Sleep targets (hours)
    min_sleep_hours: PositiveFloat = Field(default=6.0, description="Minimum healthy sleep duration")
    optimal_sleep_hours: PositiveFloat = Field(default=8.0, description="Optimal sleep duration")
    max_sleep_hours: PositiveFloat = Field(default=10.0, description="Maximum sleep before penalty")

    # Training targets (minutes)
    min_training_minutes: PositiveInt = Field(default=0, description="Minimum training minutes per day")
    optimal_training_minutes: PositiveInt = Field(default=60, description="Optimal training minutes per day")
    max_training_minutes: PositiveInt = Field(default=120, description="Maximum training before overtraining penalty")

    # Resting targets (minutes)
    min_resting_minutes: PositiveInt = Field(default=480, description="Minimum resting minutes (8 hours)")
    optimal_resting_minutes: PositiveInt = Field(default=540, description="Optimal resting minutes (9 hours)")
    max_resting_minutes: PositiveInt = Field(default=720, description="Maximum resting minutes (12 hours)")

    # Scoring method
    scoring_method: Literal["gaussian", "sigmoid", "linear"] = Field(
        default="gaussian",
        description="Method to map normalized values to scores"
    )


class ActivityScore(BaseAlgorithm):
    """
    Activity Score Algorithm - Personalized Daily Activity Assessment

    Combines multiple health metrics (steps, sleep, training, resting) into a single
    0-100 score tailored to individual user baselines and targets.

    Based on principles from Oura Ring and WHOOP band scoring algorithms.
    """

    _algorithm_name = "ActivityScore"
    _version = "v0.1.0"

    def __init__(self, settings: ActivityScoreSettings = ActivityScoreSettings()) -> None:
        self.settings = settings
        self._validate_weights()
        self.user_stats = None
        self.daily_scores = None
        self.baseline_stats = None
        return None

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0"""
        total_weight = (
            self.settings.step_weight +
            self.settings.sleep_weight +
            self.settings.training_weight +
            self.settings.resting_weight
        )
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def _compute_baseline_stats(self, baseline_data: DataFrame) -> dict:
        """
        Compute personalized baseline statistics from historical data.

        Args:
            baseline_data: DataFrame with previous N-1 days

        Returns:
            Dictionary with computed statistics
        """
        return {
            'steps_median': baseline_data['steps'].median(),
            'steps_std': baseline_data['steps'].std(),
            'steps_p75': baseline_data['steps'].quantile(0.75),
            'sleep_median': baseline_data['sleep_hours'].median(),
            'sleep_std': baseline_data['sleep_hours'].std(),
            'training_median': baseline_data['training_minutes'].median(),
            'training_std': baseline_data['training_minutes'].std(),
            'resting_median': baseline_data['resting_minutes'].median(),
            'resting_std': baseline_data['resting_minutes'].std()
        }

    def _score_steps(self, daily_steps: int, baseline_stats: dict = None) -> float:
        """
        Score daily step count (0-100).

        Gaussian distribution centered at user's median baseline.
        """
        if baseline_stats is None:
            baseline = self.settings.baseline_daily_steps
            ceiling = self.settings.step_ceiling
        else:
            # Use personalized baseline from historical data
            baseline = baseline_stats['steps_median']
            # Ceiling is 75th percentile or 1.5x median, whichever is higher
            ceiling = max(baseline * 1.5, baseline_stats['steps_p75'])

        if self.settings.scoring_method == "gaussian":
            if baseline == 0:
                return 0.0
            normalized = daily_steps / baseline
            peak_normalized = ceiling / baseline
            std_dev = (peak_normalized - 1.0) / 2.0
            score = 100 * np.exp(-((normalized - 1.0) ** 2) / (2 * std_dev ** 2))
            score = np.clip(score, 0, 100)
        else:
            if ceiling == 0:
                return 0.0
            score = min(100, (daily_steps / ceiling) * 100)

        return float(score)

    def _score_sleep(self, sleep_hours: float, baseline_stats: dict = None) -> float:
        """
        Score sleep duration (0-100).

        Centered at user's median sleep, with personalized thresholds.
        """
        if baseline_stats is None:
            optimal = self.settings.optimal_sleep_hours
            min_sleep = self.settings.min_sleep_hours
            max_sleep = self.settings.max_sleep_hours
        else:
            median_sleep = baseline_stats['sleep_median']
            std_sleep = baseline_stats['sleep_std'] or 0.5
            optimal = median_sleep
            min_sleep = max(4.0, median_sleep - std_sleep * 1.5)
            max_sleep = min(11.0, median_sleep + std_sleep * 1.5)

        if sleep_hours < min_sleep:
            deficit = (min_sleep - sleep_hours) / min_sleep
            score = max(0, 50 - deficit * 50)
        elif sleep_hours <= optimal:
            score = ((sleep_hours - min_sleep) / (optimal - min_sleep)) * 100
        elif sleep_hours <= max_sleep:
            score = 100 - ((sleep_hours - optimal) / (max_sleep - optimal)) * 30
        else:
            excess = sleep_hours - max_sleep
            score = max(0, 70 - excess * 10)

        return float(np.clip(score, 0, 100))

    def _score_training(self, training_minutes: int, baseline_stats: dict = None) -> float:
        """
        Score training time (0-100).

        Rewards consistent moderate training relative to user's baseline.
        """
        if baseline_stats is None:
            optimal = self.settings.optimal_training_minutes
            max_train = self.settings.max_training_minutes
        else:
            median_training = baseline_stats['training_median']
            std_training = baseline_stats['training_std'] or 15
            optimal = max(30, median_training)
            max_train = optimal + std_training * 2

        if training_minutes < 5:
            score = 50
        elif training_minutes <= optimal:
            score = 50 + (training_minutes / optimal) * 50
        elif training_minutes <= max_train:
            excess_ratio = (training_minutes - optimal) / (max_train - optimal)
            score = 100 + (excess_ratio * 10) - 10
        else:
            excess = training_minutes - max_train
            score = max(0, 100 - excess * 0.5)

        return float(np.clip(score, 0, 100))

    def _score_resting(self, resting_minutes: int, baseline_stats: dict = None) -> float:
        """
        Score resting/recovery time (0-100).

        Rewards recovery relative to user's baseline.
        """
        resting_hours = resting_minutes / 60

        if baseline_stats is None:
            optimal_hours = self.settings.optimal_resting_minutes / 60
            min_hours = self.settings.min_resting_minutes / 60
            max_hours = self.settings.max_resting_minutes / 60
        else:
            median_resting = baseline_stats['resting_median'] / 60
            std_resting = baseline_stats['resting_std'] / 60 or 0.5
            optimal_hours = median_resting
            min_hours = max(6.0, median_resting - std_resting * 1.5)
            max_hours = min(12.0, median_resting + std_resting * 1.5)

        if resting_hours < min_hours:
            deficit = (min_hours - resting_hours)
            score = max(0, 50 - deficit * 15)
        elif resting_hours <= optimal_hours:
            score = ((resting_hours - min_hours) / (optimal_hours - min_hours)) * 100
        elif resting_hours <= max_hours:
            excess_hours = resting_hours - optimal_hours
            max_excess = max_hours - optimal_hours
            score = 100 - (excess_hours / max_excess) * 25
        else:
            excess_hours = resting_hours - max_hours
            score = max(0, 75 - excess_hours * 10)

        return float(np.clip(score, 0, 100))

    def run(self, daily_activity_data: DataFrame):
        """
        Calculate Activity Score for the most recent day using baseline statistics.

        Args:
            daily_activity_data: DataFrame with all available data, sorted by date (oldest to newest).
                Must contain columns:
                - 'date' or index as date
                - 'steps': Daily step count
                - 'sleep_hours': Total sleep duration in hours
                - 'training_minutes': Total training time in minutes
                - 'resting_minutes': Total resting/recovery time in minutes

                Should have at least baseline_window_days of data.

        Returns:
            self with biomarker containing score for the most recent day
        """
        if not all(col in daily_activity_data.columns for col in ['steps', 'sleep_hours', 'training_minutes', 'resting_minutes']):
            raise ValueError("DataFrame must contain: steps, sleep_hours, training_minutes, resting_minutes")

        if len(daily_activity_data) < 2:
            raise ValueError(f"Need at least 2 days of data, got {len(daily_activity_data)}")

        # Ensure we have the right number of baseline days
        window_size = min(self.settings.baseline_window_days, len(daily_activity_data) - 1)

        # Split into baseline (all but last) and current day (last)
        baseline_data = daily_activity_data.iloc[:-1].tail(window_size)
        current_day_data = daily_activity_data.iloc[-1]

        # Compute personalized baseline statistics
        self.baseline_stats = self._compute_baseline_stats(baseline_data)

        # Calculate individual scores using personalized baselines
        step_score = self._score_steps(current_day_data['steps'], self.baseline_stats)
        sleep_score = self._score_sleep(current_day_data['sleep_hours'], self.baseline_stats)
        training_score = self._score_training(current_day_data['training_minutes'], self.baseline_stats)
        resting_score = self._score_resting(current_day_data['resting_minutes'], self.baseline_stats)

        # Weighted combination
        activity_score = (
            (step_score * self.settings.step_weight) +
            (sleep_score * self.settings.sleep_weight) +
            (training_score * self.settings.training_weight) +
            (resting_score * self.settings.resting_weight)
        )

        # Create output DataFrame
        score_record = {
            'date': current_day_data.get('date', daily_activity_data.index[-1]) if 'date' in daily_activity_data.columns else daily_activity_data.index[-1],
            'activity_score': round(activity_score, 1),
            'step_score': round(step_score, 1),
            'sleep_score': round(sleep_score, 1),
            'training_score': round(training_score, 1),
            'resting_score': round(resting_score, 1),
            'steps': current_day_data['steps'],
            'sleep_hours': current_day_data['sleep_hours'],
            'training_minutes': current_day_data['training_minutes'],
            'resting_minutes': current_day_data['resting_minutes'],
            'baseline_days_used': window_size
        }

        self.biomarker_agg = DataFrame([score_record])
        self.daily_scores = self.biomarker_agg.copy()

        return self

    def get_activity_score_interpretation(self, activity_score: float) -> str:
        """
        Return interpretation of Activity Score.

        Args:
            activity_score: Activity score (0-100)

        Returns:
            Interpretation string
        """
        if activity_score >= 85:
            return "Excellent - Outstanding activity and recovery balance"
        elif activity_score >= 70:
            return "Good - Healthy activity levels with adequate recovery"
        elif activity_score >= 50:
            return "Fair - Room for improvement in activity or recovery"
        elif activity_score >= 30:
            return "Poor - Significant imbalance in activity or recovery"
        else:
            return "Critical - Urgent attention needed to activity and recovery"
