# ENMO (Euclidean Norm Minus One)

## Overview

ENMO is a physical activity metric derived from accelerometer data. It represents the Euclidean norm of the acceleration vector minus 1g (gravitational acceleration), with negative values clipped to zero.

**Algorithm Name:** ENMO  
**Version:** v0.1.0

## Algorithm Description

ENMO is commonly used in wearable sensor research to quantify physical activity intensity. The calculation involves:

1. Computing the magnitude of the 3D acceleration vector: $\sqrt{x^2 + y^2 + z^2}$
2. Subtracting 1g (gravitational component): $ENMO = magnitude - 1$
3. Clipping negative values to 0: $ENMO = \max(0, magnitude - 1)$
4. Aggregating over specified time windows

## Parameters

### ENMOSettings

- **window_len** (default: 1 second) - Processing window length for calculating rolling statistics
- **aggregation_window** (default: 60 seconds) - Time window for data aggregation

## Usage Example

```python
from physiodsp.activity.enmo import ENMO, ENMOSettings
from physiodsp.sensors.imu.accelerometer import AccelerometerData
import numpy as np

# Create sample accelerometer data
timestamps = np.arange(0, 100, 0.01)  # 100 seconds at 100 Hz
x = np.random.normal(0, 0.1, len(timestamps))
y = np.random.normal(0, 0.1, len(timestamps))
z = np.ones(len(timestamps)) + np.random.normal(0, 0.1, len(timestamps))

accel_data = AccelerometerData(
    timestamps=timestamps,
    x=x,
    y=y,
    z=z,
    fs=100  # 100 Hz sampling frequency
)

# Initialize ENMO with custom settings
settings = ENMOSettings(window_len=2, aggregation_window=120)
enmo = ENMO(settings=settings)

# Run algorithm
result = enmo.run(accel_data)

# Get results
print(result.biomarker)  # DataFrame with timestamps and ENMO values

# Aggregate results
result.aggregate(method='mean')
print(result.biomarker_agg)
```

## Output

The algorithm returns a Pandas DataFrame with:

- **timestamps**: Unix timestamps of window centers
- **values**: Mean ENMO values for each window

## Clinical/Research Applications

- Quantifying daily physical activity levels
- Distinguishing sedentary vs. active periods
- Monitoring activity patterns in health studies
- Wearable device activity tracking

## References

- Accelerometer-based activity monitoring
- Physical activity quantification from wearable sensors
