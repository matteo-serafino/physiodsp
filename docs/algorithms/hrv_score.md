# HRV Score Algorithm

## Overview

The HRV Score Algorithm computes a normalized 0-100 Heart Rate Variability score based on RMSSD (Root Mean Square of Successive Differences) values. It incorporates trend analysis and stability metrics to provide a comprehensive HRV assessment.

**Algorithm Name:** HrvScoreAlgorithm  
**Version:** v0.1.0

## Algorithm Description

The HRV Score is calculated using a multi-factor approach:

### Step 1: Baseline and Variability Calculation
- **Baseline**: Median RMSSD from the baseline period (default: 30 days)
- **Standard Deviation**: Variability measure of baseline RMSSD
- **Coefficient of Variation**: $CV = \frac{std}{baseline}$

### Step 2: Z-Score Computation
$$Z = \frac{current\_rmssd - baseline}{std}$$

### Step 3: Base Score Mapping
Two methods available (default: sigmoid):
- **Sigmoid Method**: $score = 50 + 30 \times \tanh(0.5 \times Z)$
- **Percentile Method**: Maps current value to percentile of baseline distribution

### Step 4: Trend Bonus
Compares first 10 days vs. last 10 days of baseline:
- **Trend > 10%**: +10 points (improving)
- **Trend > 5%**: +5 points
- **Trend < -5%**: -5 points (declining)
- **Otherwise**: 0 points

### Step 5: Stability Penalty
Based on coefficient of variation:
- **CV > 15%**: -10 points (high variability)
- **CV > 10%**: -5 points (moderate variability)
- **Otherwise**: 0 points

### Final Score
$$HRV\_Score = \min(100, \max(0, base\_score + trend\_bonus + stability\_penalty))$$

## Parameters

### HrvScoreSettings

- **window_len** (default: 30 days) - Baseline period for comparison
- **method** (default: "sigmoid") - Scoring method: "sigmoid" or "percentile"

## Usage Example

```python
from physiodsp.hrv.hrv_score import HrvScore, HrvScoreSettings
from physiodsp.sensors.hrv import HrvData
import numpy as np

# Create sample HRV data (30+ days of daily RMSSD values)
timestamps = np.arange(0, 31)  # 31 days
rmssd_values = np.random.normal(50, 10, 31)  # RMSSD in milliseconds

hrv_data = HrvData(
    timestamps=timestamps,
    values=rmssd_values
)

# Initialize HRV Score with default settings
settings = HrvScoreSettings(window_len=30, method="sigmoid")
hrv = HrvScore(settings=settings)

# Run algorithm
result = hrv.run(hrv_data)

# Get HRV score (0-100)
print(result.biomarker_agg)
# Output:
# timestamps: 30 (current day)
# hrv_score: 65 (example score)
```

## Output

The algorithm returns a Pandas DataFrame with:

- **timestamps**: Current timestamp
- **hrv_score**: HRV score in 0-100 range

## Score Interpretation

| Score Range | Interpretation | Status |
|---|---|---|
| 80-100 | Excellent cardiovascular fitness | Very Good |
| 60-79 | Good HRV, balanced nervous system | Good |
| 40-59 | Moderate HRV | Normal |
| 20-39 | Low HRV, elevated stress | Concerning |
| 0-19 | Very low HRV, high stress | Poor |

## Key Features

### Trend Analysis
Identifies if HRV is improving or declining over time, providing actionable insights for lifestyle interventions.

### Stability Penalty
High variability in HRV measurements (indicating inconsistent data or lifestyle) reduces the score, encouraging more consistent measurement practices.

### Baseline Normalization
Uses individual baseline rather than population averages, accounting for natural individual differences in HRV.

### Flexible Scoring Methods

**Sigmoid Method**: Smooth mapping that handles outliers well
- Centered at 50 for baseline median
- Responsive to deviations from baseline

**Percentile Method**: Direct comparison to baseline distribution
- More robust to outliers
- Intuitive percentile interpretation

## Clinical Applications

- Autonomic nervous system assessment
- Recovery and readiness monitoring
- Stress level evaluation
- Cardiovascular health tracking
- Training adaptation monitoring

## HRV Background

HRV (Heart Rate Variability) represents the variation in time intervals between consecutive heartbeats. Higher HRV generally indicates:
- Better cardiovascular fitness
- Better stress management
- Better recovery
- Lower all-cause mortality risk

## Factors Affecting HRV

- **Positive Factors**: Exercise, sleep, stress management, meditation
- **Negative Factors**: Sleep deprivation, stress, illness, overtraining, poor diet

## Limitations

- Requires at least baseline window length of data
- RMSSD should be measured consistently (time of day, position, etc.)
- Environmental and lifestyle factors significantly impact scores
- Not suitable for atrial fibrillation or irregular rhythms

## References

- Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in public health, 5, 258.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065.
