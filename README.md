# PhysioDSP

A comprehensive Python library for processing and analyzing physiological sensor data. PhysioDSP provides algorithms for activity recognition, ECG and heart rate analysis, HRV (Heart Rate Variability) scoring, and digital signal processing from wearable sensors.

## Features

- **Activity Analysis**: ENMO (Euclidean Norm Minus One), personalized Activity Score, activity intensity detection, energy expenditure estimation *(coming soon)*
- **ECG Processing**: QRS peak detection, heart rate calculation, inter-beat interval analysis  
- **HRV Scoring**: Advanced heart rate variability scoring with trend analysis and stability metrics
- **Sleep Analysis**: Sleep quality and duration metrics *(coming soon)*
- **Sensor Support**: Accelerometer, gyroscope, magnetometer, and ECG data processing
- **Digital Signal Processing**: Convolution and filtering utilities for physiological signals
- **Type-Safe Design**: Built with Pydantic for robust data validation
- **Configurable Algorithms**: Each algorithm includes customizable settings and parameters

## Installation

```bash
pip install physiodsp
```

### Requirements

- Python >= 3.11
- NumPy >= 2.4.0
- Pandas >= 2.3.3
- SciPy >= 1.16.3
- Pydantic >= 2.12.0

## Quick Start

### Activity Analysis - ENMO

```python
from physiodsp.activity.enmo import ENMO, ENMOSettings
from physiodsp.sensors.imu.accelerometer import AccelerometerData

# Create accelerometer data
accel_data = AccelerometerData(
    timestamps=timestamps,
    x=x_values,
    y=y_values,
    z=z_values,
    fs=64  # 64 Hz sampling frequency
)

# Initialize and run ENMO algorithm
enmo = ENMO(settings=ENMOSettings(window_len=1, aggregation_window=60))
result = enmo.run(accel_data)

# Get results
print(result.biomarker)  # DataFrame with timestamps and ENMO values
result.aggregate(method='mean')  # Aggregate results
```

### ECG Peak Detection

```python
from physiodsp.ecg.peak_detector import EcgPeakDetector
from physiodsp.sensors.ecg import EcgData

# Create ECG data
ecg_data = EcgData(
    timestamps=timestamps,
    values=ecg_values,
    fs=250  # 250 Hz sampling frequency
)

# Initialize and run peak detector
detector = EcgPeakDetector()
result = detector.run(ecg_data)

# Get heart rate and inter-beat intervals
print(result.biomarker)  # DataFrame with RR intervals and heart rate
```

### HRV Scoring

```python
from physiodsp.hrv.hrv_score import HrvScore, HrvScoreSettings
from physiodsp.sensors.hrv import HrvData

# Create HRV data (RMSSD values)
hrv_data = HrvData(
    timestamps=timestamps,
    values=rmssd_values
)

# Calculate HRV score
hrv = HrvScore(settings=HrvScoreSettings(window_len=30, method="sigmoid"))
result = hrv.run(hrv_data)

# Get HRV score (0-100)
print(result.biomarker_agg)  # DataFrame with HRV score
```

## Core Modules

### `activity/`
- **ENMO** ✅ (unit tested): Euclidean Norm Minus One - physical activity metric from accelerometer
- **Zero Crossing** ✅ (unit tested): Activity intensity detection
- **Time Above Threshold** ✅ (unit tested): Vigorous activity quantification
- **PIM** ✅ (unit tested): Proportional Integration Mode - multi-axis activity processing
- **Activity Score** ✅ (unit tested): Personalized 0-100 daily activity and recovery score with baseline personalization
- **Energy Expenditure** (coming soon): Calorie burn estimation algorithms
- **Activity Recognition** (coming soon): Machine learning-based activity classification
- **Sleep Metrics** (coming soon): Sleep quality and duration analysis

### `sensors/`
- **IMU (Inertial Measurement Unit)**: 
  - Accelerometer data processing
  - Gyroscope data processing
  - Magnetometer data processing
- **ECG**: Electrocardiogram signal handling
- **HRV**: Heart rate variability metrics

### `ecg/`
- **Peak Detector**: QRS complex detection using filtering and peak detection

### `hrv/`
- **HRV Score**: Comprehensive 0-100 HRV scoring with trend and stability analysis

### `dsp/`
- **Convolution**: Signal processing utilities including moving averages

## Algorithm Architecture

All algorithms follow a consistent architecture:

1. **Settings**: Pydantic-based configuration models for algorithm parameters
2. **Algorithm Class**: Inherits from `BaseAlgorithm` with a `run()` method
3. **Data Models**: Type-safe sensor data classes with validation
4. **Results**: Biomarkers stored in Pandas DataFrames for easy analysis

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install test dependencies
pip install -r requirements-test.txt
```

### Running Tests

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing project structure
- All new algorithms inherit from `BaseAlgorithm`
- Settings are defined using Pydantic models
- Tests are included for new functionality

## License

[Add your license here]

## References

- ENMO algorithm: based on accelerometer magnitude data
- ECG Peak Detection: Pan-Tompkins-like approach with filtering
- HRV Scoring: Z-score based with trend and stability analysis
