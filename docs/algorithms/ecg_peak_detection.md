# ECG Peak Detector Algorithm

## Overview

The ECG Peak Detector is a QRS complex detection algorithm that identifies R-peaks in electrocardiogram signals. It automatically extracts heart rate and inter-beat intervals from raw ECG data.

**Algorithm Name:** EcgPeakDetector  
**Version:** v0.1.0

## Algorithm Description

This algorithm implements a Pan-Tompkins-like approach for robust QRS detection:

1. **Band-pass Filtering**: Filters ECG signal between 5-15 Hz to enhance QRS complexes
2. **Derivative Filtering**: Applies a 5-tap derivative filter to amplify signal changes
3. **Signal Squaring**: Squares the derivative output to emphasize peaks
4. **Moving Average**: Applies moving average filter (window: 0.15 seconds) for smoothing
5. **Peak Detection**: Identifies peaks using adaptive thresholding
6. **RR Interval Calculation**: Computes inter-beat intervals and heart rate

## Parameters

- **FILTER_ORDER**: 3 (Butterworth filter order)
- **LOWER_F_CUT**: 5 Hz (lower cutoff frequency)
- **UPPER_F_CUT**: 15 Hz (upper cutoff frequency)
- **WIN_LEN_SEC**: 0.150 seconds (moving average window)

## Usage Example

```python
from physiodsp.ecg.peak_detector import EcgPeakDetector
from physiodsp.sensors.ecg import EcgData
import numpy as np

# Create sample ECG data (250 Hz sampling rate)
timestamps = np.arange(0, 60, 1/250)  # 60 seconds at 250 Hz
# Simulate ECG signal with 60 bpm heart rate
heart_rate = 60
t = timestamps
ecg_signal = (
    0.5 * np.sin(2 * np.pi * (heart_rate/60) * t) +
    0.2 * np.sin(2 * np.pi * (2*heart_rate/60) * t) +
    0.1 * np.sin(2 * np.pi * (3*heart_rate/60) * t)
)

ecg_data = EcgData(
    timestamps=timestamps,
    values=ecg_signal,
    fs=250  # 250 Hz sampling frequency
)

# Initialize peak detector
detector = EcgPeakDetector()

# Run algorithm
result = detector.run(ecg_data)

# Get results
print(result.biomarker)
# DataFrame columns:
# - timestamps: Detected R-peak timestamps
# - inter_beat_interval: RR intervals in milliseconds
# - heart_rate: Instantaneous heart rate in bpm
```

## Output

The algorithm returns a Pandas DataFrame with:

- **timestamps**: Time points of detected R-peaks
- **inter_beat_interval**: RR intervals in milliseconds
- **heart_rate**: Instantaneous heart rate in beats per minute (bpm)

## Signal Processing Steps Explained

### 1. Band-pass Filtering
Removes baseline wander (<5 Hz) and noise (>15 Hz), preserving QRS characteristics.

### 2. Derivative Filtering
Enhances the steep slopes of QRS complexes:
$$h[n] = [-1/8, -2/8, 0, 2/8, 1/8]$$

### 3. Squaring
Nonlinear operation to emphasize peaks: $y[n] = (x[n])^2$

### 4. Moving Average
Smoothing filter with window of 0.15 seconds at sampling rate.

### 5. Peak Detection
Adaptive threshold based on mean signal level, with minimum distance constraint to avoid detecting multiple peaks within same QRS.

## Clinical Applications

- Continuous heart rate monitoring
- Arrhythmia detection
- Exercise intensity assessment
- Sleep quality monitoring
- ECG signal quality validation

## Important Notes

- Optimal for ECG signals sampled at 250 Hz or higher
- Assumes relatively clean ECG signals
- May struggle with severe noise or motion artifacts
- Best used with proper electrode placement

## Limitations

- Sensitive to baseline wander if pre-filtering is inadequate
- May miss peaks during high-frequency noise
- Assumes relatively regular heart rhythms for optimal performance
- Requires adequate ECG signal quality

## References

- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE transactions on biomedical engineering, (3), 230-236.
- Accelerometer-based QRS detection variations
