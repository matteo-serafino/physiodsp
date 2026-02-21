# PIM (Proportional Integration Mode) Algorithm

## Overview

The PIM (Proportional Integration Mode) Algorithm processes 3-axis IMU (Inertial Measurement Unit) data to extract activity metrics. It computes the absolute values of acceleration on each axis and aggregates them over time windows.

**Algorithm Name:** PIMAlgorithm  
**Version:** v0.1.0

## Algorithm Description

PIM is a signal processing technique that:

1. Computes absolute acceleration values for X, Y, and Z axes
2. Aggregates data over time windows using the specified method
3. Provides multi-axis activity representation

This approach preserves directional information while quantifying movement intensity across all three spatial dimensions.

## Parameters

- **Aggregation Window**: Default 5 seconds
- **Aggregation Method**: Sum (default) or other statistical measures (mean, max, etc.)

## Usage Example

```python
from physiodsp.activity.pim import PIMAlgorithm
from physiodsp.sensors.imu.base import IMUData
import numpy as np

# Create sample IMU data
timestamps = np.arange(0, 100, 0.01)  # 100 seconds at 100 Hz
x = np.random.normal(0, 0.2, len(timestamps))
y = np.random.normal(0, 0.2, len(timestamps))
z = np.ones(len(timestamps)) + np.random.normal(0, 0.2, len(timestamps))

imu_data = IMUData(
    timestamps=timestamps,
    x=x,
    y=y,
    z=z,
    fs=100  # 100 Hz sampling frequency
)

# Initialize PIM algorithm
pim = PIMAlgorithm()

# Estimate activity
result = pim.estimate(imu_data)

# Aggregate results
result.aggregate(method='sum')

# Get aggregated results
print(result.biomarker_agg)
```

## Output

The algorithm returns a Pandas DataFrame with:

- **timestamps_unix**: Aggregated time windows (unix timestamps)
- **x**: Aggregated absolute acceleration on X-axis
- **y**: Aggregated absolute acceleration on Y-axis
- **z**: Aggregated absolute acceleration on Z-axis

## Applications

- Multi-axis activity monitoring
- Movement magnitude assessment
- Directional activity analysis
- Device orientation-independent activity tracking

## Advantages

- Preserves directional information
- Resistant to device orientation changes
- Suitable for complex movement patterns
- Useful for rehabilitation and sports analytics

## References

- Proportional Integration algorithms in signal processing
- Multi-axis accelerometer data analysis
