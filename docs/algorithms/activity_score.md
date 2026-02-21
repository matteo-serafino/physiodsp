# Activity Score Algorithm

## Overview

The Activity Score Algorithm provides a personalized 0-100 daily activity and recovery score based on steps, sleep, training, and resting metrics. It uses individual baseline data to personalize scoring, accounting for natural variations in activity levels and sleep patterns across different users.

**Algorithm Name:** ActivityScore  
**Version:** v0.1.0

## Algorithm Description

The Activity Score is calculated using a multi-factor personalization approach:

### Step 1: Baseline Statistics Calculation
Using historical data (default: 30 days prior to current day):
- **Steps**: Median, standard deviation, 25th and 75th percentiles
- **Sleep Hours**: Median, standard deviation, 25th and 75th percentiles
- **Training Minutes**: Median, standard deviation, 25th and 75th percentiles
- **Resting Minutes**: Median, standard deviation, 25th and 75th percentiles

### Step 2: Individual Metric Scoring
Each metric is scored 0-100 based on baseline-relative thresholds:

#### Steps Scoring
- **Excellent** (≥95): Steps ≥ 75th percentile + 1 std
- **Very Good** (≥85): Steps ≥ 75th percentile
- **Good** (≥70): Steps ≥ median
- **Fair** (≥50): Steps ≥ 25th percentile
- **Poor**: Steps < 25th percentile

#### Sleep Hours Scoring
- **Excellent** (≥95): Sleep ≥ 75th percentile + 0.5 std
- **Very Good** (≥85): Sleep ≥ 75th percentile
- **Good** (≥70): Sleep ≥ median
- **Fair** (≥50): Sleep ≥ 25th percentile - 0.5 std
- **Poor**: Sleep < 25th percentile - 0.5 std

#### Training Minutes Scoring
- **Excellent** (≥95): Training ≥ 75th percentile + 1 std
- **Very Good** (≥85): Training ≥ 75th percentile
- **Good** (≥70): Training ≥ median
- **Fair** (≥50): Training ≥ 25th percentile
- **Poor**: Training < 25th percentile

#### Resting Minutes Scoring
- **Excellent** (≥95): Resting ≥ 75th percentile
- **Very Good** (≥85): Resting ≥ median
- **Good** (≥70): Resting ≥ 25th percentile
- **Fair** (≥50): Resting ≥ 10th percentile
- **Poor**: Resting < 10th percentile

### Step 3: Weighted Combination
$$ActivityScore = (steps\_score × 0.25) + (sleep\_score × 0.35) + (training\_score × 0.25) + (resting\_score × 0.15)$$

Final score is clamped to 0-100 range.

## Parameters

### ActivityScoreSettings

- **baseline_window_days** (default: 30) - Number of prior days used for baseline calculation
- **step_weight** (default: 0.25) - Weight for steps component
- **sleep_weight** (default: 0.35) - Weight for sleep component
- **training_weight** (default: 0.25) - Weight for training component
- **resting_weight** (default: 0.15) - Weight for resting component

## Usage Example

```python
from physiodsp.activity.activity_score import ActivityScore, ActivityScoreSettings
from pandas import DataFrame
from datetime import datetime, timedelta

# Create 31 days of activity data (30-day baseline + 1 current day)
dates = [datetime.now() - timedelta(days=x) for x in range(30, -1, -1)]
data = DataFrame({
    'date': dates,
    'steps': [8500] * 31,
    'sleep_hours': [7.5] * 31,
    'training_minutes': [45] * 31,
    'resting_minutes': [500] * 31
})

# Initialize Activity Score with default settings
settings = ActivityScoreSettings(baseline_window_days=30)
algorithm = ActivityScore(settings=settings)

# Run algorithm
result = algorithm.run(data)

# Get activity score and component scores
print(result.biomarker_agg)
# Output:
# activity_score: 72
# step_score: 70
# sleep_score: 75
# training_score: 68
# resting_score: 70
# baseline_days_used: 30
```

## Output

The algorithm returns a result object with:

### biomarker_agg (DataFrame)
- **activity_score**: Overall activity score (0-100)
- **step_score**: Steps component score (0-100)
- **sleep_score**: Sleep component score (0-100)
- **training_score**: Training component score (0-100)
- **resting_score**: Resting component score (0-100)
- **baseline_days_used**: Number of baseline days used for calculation

### baseline_stats (Dict)
- **steps_median**, **steps_std**, **steps_p25**, **steps_p75**
- **sleep_median**, **sleep_std**, **sleep_p25**, **sleep_p75**
- **training_median**, **training_std**, **training_p25**, **training_p75**
- **resting_median**, **resting_std**, **resting_p25**, **resting_p75**

## Score Interpretation

| Score Range | Interpretation | Recommendation |
|---|---|---|
| ≥85 | Excellent - Outstanding activity and recovery balance | Maintain current routine |
| 70-84 | Good - Healthy activity levels with adequate recovery | Continue current habits |
| 50-69 | Fair - Room for improvement in activity or recovery | Increase activity or improve sleep |
| 30-49 | Poor - Significant imbalance in activity or recovery | Action needed for health |
| <30 | Critical - Urgent attention needed to activity and recovery | Consult healthcare provider |

## Key Features

### Personalized Baseline
Uses individual baseline rather than population averages, accounting for:
- Natural variations in activity levels
- Age and fitness level differences
- Individual sleep needs
- Work and lifestyle schedules

### Balanced Weighting
- **Sleep (35%)**: Highest weight, reflecting critical importance for recovery
- **Steps (25%)** + **Training (25%)**: Equal activity components
- **Resting (15%)**: Recovery quality metrics

### Multi-Component Analysis
Evaluates four independent dimensions:
- **Steps**: Daily physical activity volume
- **Sleep**: Nighttime recovery quality
- **Training**: Structured exercise intensity
- **Resting**: Daytime recovery and low-activity periods

### Trend-Aware
Automatically adapts to individual patterns by using rolling baseline calculation.

## Practical Applications

- **Fitness Tracking**: Monitor daily activity and recovery balance
- **Athlete Monitoring**: Track training load and recovery
- **Health Management**: Identify patterns affecting wellbeing
- **Behavior Change**: Personalized feedback on activity goals
- **Occupational Health**: Monitor activity levels in sedentary jobs
- **Sleep Optimization**: Balance activity with adequate rest

## Factors Affecting Score

### Positive Factors
- Consistent physical activity (8,000-12,000+ steps/day)
- Adequate sleep (7-8.5 hours/night)
- Regular structured training
- Sufficient rest and recovery time

### Negative Factors
- Sedentary behavior (<5,000 steps/day)
- Sleep deprivation (<6 hours/night)
- Overtraining without recovery
- Excessive resting or low activity

## Minimum Data Requirements

- **Minimum**: At least 2 days of data (1 baseline day + 1 evaluation day)
- **Recommended**: At least 30 days for stable baseline
- **Optimal**: 60-90 days for seasonal pattern detection

## Limitations

- Requires consistent daily data logging
- Does not account for activity intensity beyond training minutes
- Not suitable for irregular schedules (shift workers without adjustment)
- Sleep quality not measured (duration only)
- Does not detect non-wear periods or data gaps automatically
- Baseline calculation requires sufficient historical data

## Algorithm Parameters Customization

```python
# Custom weighting emphasizing sleep
custom_settings = ActivityScoreSettings(
    baseline_window_days=30,
    step_weight=0.20,
    sleep_weight=0.45,
    training_weight=0.20,
    resting_weight=0.15
)
algorithm = ActivityScore(settings=custom_settings)

# Short-term baseline (last 15 days)
short_baseline = ActivityScoreSettings(baseline_window_days=15)
algorithm = ActivityScore(settings=short_baseline)
```

## Related Metrics

**Complementary Metrics**:
- **ENMO (Euclidean Norm Minus One)**: Raw accelerometer-based activity metric
- **HRV Score**: Heart rate variability for autonomic nervous system assessment
- **ECG Analysis**: Heart rate and cardiac metrics

## References

- Wearable activity data standards and metrics
- Sleep hygiene research and recommendations
- Training load and recovery models in sports science
- Personalized health monitoring approaches

## Version History

- **v0.1.0** (2026-02-21): Initial alpha release with 30-day baseline personalization
