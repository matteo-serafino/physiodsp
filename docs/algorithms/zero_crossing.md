# Zero Crossing Algorithm

## Overview

The Zero Crossing algorithm detects activity by counting the number of times the acceleration signal crosses zero within a specified threshold. This metric is useful for identifying movement patterns and activity intensity from accelerometer data.

**Algorithm Name:** ZeroCrossingAlgorithm  
**Version:** v0.1.0

## Algorithm Description

Zero crossing rate (ZCR) is a common feature in signal processing to detect signal changes and activity transitions. The algorithm:

1. Computes differences between consecutive acceleration samples
2. Identifies zero crossings (sign changes) exceeding a threshold
3. Counts crossings within sliding windows
4. Aggregates results over specified time periods

Zero crossing rate increases with activity intensity, making it useful for activity classification.

## Parameters

### ZeroCrossingSettings

- **window_len** (default: 1 second) - Processing window length for calculating zero crossing rate
- **aggregation_window** (default: 60 seconds) - Time window for data aggregation
- **zero_crossing_thr** (default: 0.05 g) - Threshold for detecting zero crossings

## Usage Example

```python
from physiodsp.activity.zero_crossing import ZeroCrossing, ZeroCrossingSettings
from physiodsp.sensors.imu.base import IMUData
import numpy as np

# Create sample IMU data
timestamps = np.arange(0, 100, 0.01)  # 100 seconds at 100 Hz
x = np.sin(np.linspace(0, 20*np.pi, len(timestamps))) * 0.5
y = np.cos(np.linspace(0, 20*np.pi, len(timestamps))) * 0.5
z = np.ones(len(timestamps))

imu_data = IMUData(
    timestamps=timestamps,
    x=x,
    y=y,
    z=z,
    fs=100  # 100 Hz sampling frequency
)

# Initialize Zero Crossing with custom settings
settings = ZeroCrossingSettings(
    window_len=1,
    aggregation_window=60,
    zero_crossing_thr=0.05
)
zcr = ZeroCrossing(settings=settings)

# Run algorithm
result = zcr.run(imu_data)

# Get results
print(result.biomarker)  # DataFrame with zero crossing counts
```

## Output

The algorithm returns a Pandas DataFrame with:

- **timestamps**: Unix timestamps of window centers
- **x**: Zero crossing rate for X-axis
- **y**: Zero crossing rate for Y-axis
- **z**: Zero crossing rate for Z-axis

## Applications

- Activity intensity estimation
- Movement pattern recognition
- Wake/sleep cycle detection
- Tremor and movement disorder assessment

## References

- Zero crossing rate as a signal feature
- Accelerometer-based activity recognition
