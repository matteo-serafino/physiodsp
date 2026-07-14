# Step Count Algorithm

## Overview

The Step Count algorithm provides a robust mechanism for counting steps and estimating gait cadence from wrist-worn accelerometer data. It employs a multi-stage pipeline designed to distinguish intentional locomotion from random movement and noise, ensuring high accuracy across various activity intensities (from slow walking to running).

**Algorithm Name:** StepCount  
**Version:** 0.1.0

## Algorithm Description

The algorithm processes tri-axial acceleration data through a five-stage pipeline:

### 1. Pre-processing
- **Vector Magnitude (VM)**: Computes the Euclidean norm of the 3D acceleration vector ($\sqrt{x^2 + y^2 + z^2}$) to ensure the algorithm is orientation-independent.
- **Gravity Removal**: Applies a high-pass Butterworth filter to remove the 1g DC component and slow drifts, leaving only the dynamic acceleration signal.

### 2. Motion Gate
To prevent false positives during sedentary periods, the signal is passed through a motion gate:
- **SMA Thresholding**: Calculates the Signal Magnitude Area (SMA).
- **Classification**: Windows with SMA below a specific threshold are classified as "sleep" or "rest" and are excluded from step counting.

### 3. Periodicity Detection
Locomotion is characterized by periodic signals. The algorithm confirms locomotion using a dual-verification approach:
- **Autocorrelation**: Analyzes the signal's self-similarity to find a dominant peak within the expected gait frequency range (0.5 Hz to 3.5 Hz).
- **PSD Purity**: If the autocorrelation peak is ambiguous, the algorithm computes the Power Spectral Density (PSD) and checks the peak-to-mean ratio (spectral purity) to confirm a rhythmic gait pattern.

### 4. Peak Detection
Once locomotion is confirmed, steps are detected using an adaptive approach:
- **Adaptive Band-pass Filtering**: Filters the signal around the detected cadence frequency to isolate the stepping component.
- **Adaptive Thresholding**: Steps are identified as peaks that exceed a threshold proportional to the local RMS amplitude, allowing the algorithm to remain accurate regardless of whether the user is walking softly or running vigorously.

### 5. Validation and Post-processing
The raw step detections are refined to remove artifacts:
- **Bout Grouping**: Consecutive steps are grouped into "bouts" of activity.
- **Short-Bout Filtering**: Bouts containing fewer than a minimum number of steps (default: 3) are discarded as noise.
- **Bout Merging**: Bouts separated by a short time gap are merged into a single continuous activity period.
- **Step Interpolation**: Small gaps in detection (e.g., a single missed peak) are filled based on the prevailing cadence to improve count accuracy.

## Parameters

### StepCountSettings
- **hp_cutoff_hz** (default: 0.5 Hz) - Cutoff frequency for the high-pass filter to remove gravity.
- **sma_sleep_threshold** (default: 0.02 g) - SMA below this value classifies the window as sleep/rest.
- **sma_adl_threshold** (default: 0.05 g) - SMA above this value is treated as clear locomotion energy.
- **periodicity_threshold** (default: 0.50) - Base autocorrelation peak threshold for locomotion classification.
- **spectral_purity_threshold** (default: 8.0) - PSD peak-to-mean ratio used for secondary confirmation.
- **threshold_factor** (default: 0.30) - Factor used to set the adaptive peak threshold relative to local RMS.
- **analysis_window_s** (default: 2.0 s) - Processing window length for each segment.
- **min_bout_steps** (default: 3) - Minimum steps required for a bout to be retained.
- **bout_merge_gap_s** (default: 3.0 s) - Maximum gap between bouts to trigger a merge.
- **interpolation_max_void_s** (default: 2.0 s) - Maximum gap duration for missing step interpolation.
- **aggregation_window** (default: 60 s) - Time window for data aggregation in `aggregate()`.

## Usage Example

```python
from physiodsp.activity.step_count import StepCount, StepCountSettings
from physiodsp.sensors.imu.accelerometer import AccelerometerData
import numpy as np

# Create sample accelerometer data (100 Hz)
fs = 100
duration = 60
t = np.arange(0, duration, 1/fs)
# Simulate a 1.6Hz gait signal on Z axis with gravity
z = 1.0 + 0.4 * np.sin(2 * np.pi * 1.6 * t) 
x = np.random.normal(0, 0.05, len(t))
y = np.random.normal(0, 0.05, len(t))

accel_data = AccelerometerData(
    timestamps=t, x=x, y=y, z=z, fs=fs
)

# Initialize StepCount algorithm
settings = StepCountSettings(bout_merge_gap_s=5.0)
sc = StepCount(settings=settings)

# Run algorithm
result = sc.run(accel_data)

# Get total steps and biomarker DataFrame
print(f"Total steps: {result.total_steps}")
print(result.biomarker.head())

# Aggregate results (e.g., mean cadence per minute)
result.aggregate(method='mean')
print(result.biomarker_agg)
```

## Output

The algorithm provides two primary outputs:

1. **Total Steps**: An integer representing the total count of validated steps.
2. **Biomarker DataFrame**: A Pandas DataFrame containing:
    - **timestamps**: Time markers for each aggregation bin.
    - **step_count**: Cumulative step count over time.
    - **cadence_spm**: Steps per minute (SPM) for that window.
    - **activity_state**: Classification of the activity ("walk", "run", "rest", or "sleep").

## Applications

- **Gait Analysis**: Estimating cadence and identifying activity transitions.
- **Physical Activity Tracking**: Quantifying total daily steps.
- **Health Monitoring**: Assessing movement patterns in clinical populations.
- **Activity Classification**: Distinguishing between different modes of locomotion (walking vs. running).
