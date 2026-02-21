# Time Above Threshold Algorithm

## Overview

The Time Above Threshold algorithm measures the duration and frequency of periods when acceleration exceeds a specified threshold. This metric is useful for identifying active periods and activity intensity levels from accelerometer data.

**Algorithm Name:** TimeAboveThrAlgorithm  
**Version:** v0.1.0

## Algorithm Description

This algorithm detects periods of elevated acceleration that exceed a user-defined threshold. It:

1. Computes absolute acceleration values for each axis
2. Identifies samples exceeding the threshold
3. Calculates time spent above threshold within windows
4. Aggregates results over specified time periods

Time above threshold can indicate intensity of physical activity, with higher values suggesting more intense or vigorous activity.

## Parameters

### TimeAboveThrSettings

- **window_len** (default: 1 second) - Processing window length for calculating time above threshold
- **aggregation_window** (default: 60 seconds) - Time window for data aggregation
- **threshold** (default: 0.1 g) - Acceleration threshold in gravitational units (g)

## Usage Example

```python
from physiodsp.activity.time_above_thr import TimeAboveThr, TimeAboveThrSettings
from physiodsp.sensors.imu.base import IMUData
import numpy as np

# Create sample IMU data
timestamps = np.arange(0, 100, 0.01)  # 100 seconds at 100 Hz
x = np.random.uniform(-0.5, 0.5, len(timestamps))
y = np.random.uniform(-0.5, 0.5, len(timestamps))
z = np.ones(len(timestamps))

imu_data = IMUData(
    timestamps=timestamps,
    x=x,
    y=y,
    z=z,
    fs=100  # 100 Hz sampling frequency
)

# Initialize Time Above Threshold with custom settings
settings = TimeAboveThrSettings(
    window_len=1,
    aggregation_window=60,
    threshold=0.2  # 0.2g threshold
)
tat = TimeAboveThr(settings=settings)

# Run algorithm
result = tat.run(imu_data)

# Get results
print(result.biomarker)  # DataFrame with time above threshold values
```

## Output

The algorithm returns a Pandas DataFrame with:

- **timestamps**: Unix timestamps of window centers
- **x**: Time (or count) of X-axis samples above threshold
- **y**: Time (or count) of Y-axis samples above threshold
- **z**: Time (or count) of Z-axis samples above threshold

## Applications

- Vigorous activity detection
- Activity intensity classification
- Movement pattern monitoring
- Rehabilitation progress tracking
- Physical therapy compliance

## Clinical Significance

Different thresholds can identify different activity intensities:
- Lower thresholds (0.05-0.1g): Light activity
- Medium thresholds (0.2-0.5g): Moderate activity
- Higher thresholds (>0.5g): Vigorous activity

## References

- Accelerometer-based activity intensity measurement
- Threshold-based activity recognition
