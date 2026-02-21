# PhysioDSP

A comprehensive Python library for processing and analyzing physiological sensor data from wearable devices.

## What is PhysioDSP?

PhysioDSP is a specialized digital signal processing library designed for physiological monitoring applications. It provides state-of-the-art algorithms for:

- **Activity Recognition & Quantification**: Extract meaningful activity metrics from accelerometer data
- **Cardiac Analysis**: Advanced ECG processing and heart rate variability assessment
- **Sensor Data Processing**: Robust handling of IMU, ECG, and HRV sensor data
- **Biomarker Extraction**: Generate clinically-relevant health metrics from raw sensor signals

Whether you're building a fitness tracker, health monitoring application, or conducting physiological research, PhysioDSP provides the algorithms you need.

## Key Features

### 🎯 Activity Analysis
- **ENMO**: Euclidean Norm Minus One - standard physical activity metric
- **Zero Crossing**: Activity intensity detection
- **Time Above Threshold**: Vigorous activity quantification
- **PIM**: Multi-axis activity processing

### ❤️ Cardiac Analysis
- **ECG Peak Detection**: Automatic QRS complex detection and heart rate extraction
- **HRV Score**: Comprehensive 0-100 heart rate variability assessment with trend analysis

### 🔧 Type-Safe Design
- Built with Pydantic for robust data validation
- Consistent API across all algorithms
- Type hints for better IDE support

### ⚙️ Configurable
- Customizable algorithm parameters via settings classes
- Flexible aggregation methods
- Support for various data formats

## Quick Start

### Installation

```bash
pip install physiodsp
```

### Basic Usage

```python
from physiodsp.activity.enmo import ENMO, ENMOSettings
from physiodsp.sensors.imu.accelerometer import AccelerometerData

# Prepare your accelerometer data
accel_data = AccelerometerData(
    timestamps=timestamps,
    x=x_values,
    y=y_values,
    z=z_values,
    fs=64  # 64 Hz sampling frequency
)

# Run ENMO algorithm
enmo = ENMO(settings=ENMOSettings(window_len=1, aggregation_window=60))
result = enmo.run(accel_data)

# Access results
print(result.biomarker)  # Pandas DataFrame
```

## Available Algorithms

### Activity Algorithms

| Algorithm | Purpose | Input | Output |
|-----------|---------|-------|--------|
| [ENMO](algorithms/enmo.md) | Physical activity intensity | Accelerometer | Activity values |
| [Zero Crossing](algorithms/zero_crossing.md) | Movement frequency detection | Accelerometer | Zero crossing counts |
| [Time Above Threshold](algorithms/time_above_threshold.md) | Vigorous activity duration | Accelerometer | Time above threshold |
| [PIM](algorithms/pim.md) | Multi-axis activity | IMU (3-axis) | Per-axis activity |

### Cardiac Algorithms

| Algorithm | Purpose | Input | Output |
|-----------|---------|-------|--------|
| [ECG Peak Detection](algorithms/ecg_peak_detection.md) | Heart rate extraction | ECG signal | HR, RR intervals |
| [HRV Score](algorithms/hrv_score.md) | Cardiovascular health | RMSSD values | HRV score (0-100) |

## Algorithm Design

All algorithms in PhysioDSP follow a consistent, intuitive design pattern:

```python
# 1. Define settings (optional, sensible defaults provided)
settings = MyAlgorithmSettings(param1=value1, param2=value2)

# 2. Initialize algorithm
algorithm = MyAlgorithm(settings=settings)

# 3. Run on sensor data
result = algorithm.run(sensor_data)

# 4. Access results
print(result.biomarker)  # Pandas DataFrame with timestamps and values
```

All results are returned as Pandas DataFrames for easy integration with data analysis workflows.

## Use Cases

### 📊 Fitness & Wellness
- Daily activity tracking and quantification
- Exercise intensity monitoring
- Recovery assessment via HRV

### 🏥 Clinical Research
- Activity pattern analysis in patient populations
- Cardiac health monitoring
- Rehabilitation progress tracking

### 🔬 Wearable Development
- Activity-based user feedback
- Sleep detection and analysis
- Real-time health monitoring

### 📱 Mobile Applications
- Personalized health dashboards
- Trend analysis and insights
- User engagement metrics

## Data Types Supported

- **Accelerometer**: 3-axis acceleration data (typically 20-100 Hz)
- **Gyroscope**: 3-axis rotation rate (optional, for advanced analysis)
- **Magnetometer**: 3-axis magnetic field (optional, for orientation)
- **ECG**: Single-channel electrocardiogram (typically 250+ Hz)
- **HRV**: Aggregated RMSSD values

## Documentation Structure

- **Home** (this page): Overview and quick start
- **About**: Project information and vision
- **Algorithms**: Detailed documentation for each algorithm
  - Algorithm description and mathematics
  - Parameter explanations
  - Usage examples
  - Clinical applications
  - Performance considerations

## Requirements

- Python >= 3.11
- NumPy >= 2.4.0
- Pandas >= 2.3.3
- SciPy >= 1.16.3
- Pydantic >= 2.12.0

## Getting Help

- Check the algorithm-specific documentation for detailed examples
- Review the parameter descriptions in each algorithm's settings class
- Examine the examples in the `tests/` directory for real-world usage patterns

## Contributing

Contributions are welcome! If you'd like to:
- Add new algorithms
- Improve existing implementations
- Report bugs
- Suggest enhancements

Please check the project repository for contribution guidelines.

## License

[License information to be added]

## Acknowledgments

This library draws inspiration from established signal processing techniques and physiological monitoring best practices in research and clinical settings.

---

**Ready to get started?** Pick an algorithm from the [Algorithms section](algorithms/enmo.md) and follow the usage examples, or jump to the [About page](about.md) to learn more about the project.