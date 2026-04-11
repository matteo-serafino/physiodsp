# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-12

### Added

#### Balance Tests
- **Sway**: Postural sway analysis from accelerometer data during balance tests
  - Medio-lateral (ML) displacement derived from the X axis, antero-posterior (AP) from the Z axis
  - Small-angle approximation: displacement = acceleration × sensor height
  - Zero-phase Butterworth low-pass filtering (default: order 4, cutoff 2.5 Hz)
  - Mean-centred displacement signals for bias removal
  - Stabilometric indices for full path, ML path, and AP path:
    - Average distance, RMS distance, total path length, average velocity
  - 95% confidence ellipse via SVD of the ML–AP covariance matrix (area, angle, semi-axes)
  - `SwaySettings`: configurable filter order and cutoff frequency
  - Unit tests in `test_balance_tests_sway.py`
  - Algorithm documentation added in `docs/algorithms/sway.md`
  - MkDocs navigation updated with a new *Balance Tests* section

### Changed

#### Apply semantic versioning
- Apply semantic versioning to all the algorithms

---

## [0.1.0-beta] - 2026-02-21

### Added

#### Core Features
- Initial alpha release of PhysioDSP library
- Type-safe design with Pydantic-based settings classes
- Consistent algorithm API for all signal processing modules
- Support for multiple sensor types: Accelerometer, Gyroscope, Magnetometer, ECG, HRV

#### Activity Analysis Algorithms
- **ENMO (Euclidean Norm Minus One)**: Physical activity metric from 3-axis accelerometer data
  - Configurable window length and aggregation window
  - Per-sample and aggregated biomarker extraction
- **Zero Crossing Algorithm**: Activity frequency detection using zero crossing rate
  - Per-axis zero crossing counting
  - Configurable threshold and window parameters
- **Time Above Threshold**: Vigorous activity quantification
  - Detects time spent above acceleration threshold
  - Multi-axis analysis with independent thresholds
- **PIM (Proportional Integration Mode)**: Multi-axis activity processing
  - Absolute value computation across three axes
  - Aggregation over configurable time windows
- **Activity Score**: Personalized daily activity and recovery scoring (0-100)
  - Baseline personalization using historical data (default: 30 days)
  - Multi-component scoring: steps, sleep, training, training minutes, resting minutes
  - Adaptive weighting with customizable parameters
  - Statistical baseline calculation: median, std dev, percentiles
  - Individual metric scoring based on percentile thresholds
  - Score interpretations: Excellent (≥85), Good (70-84), Fair (50-69), Poor (30-49), Critical (<30)

#### Cardiac Analysis Algorithms
- **ECG Peak Detector**: QRS complex detection using Pan-Tompkins approach
  - Bandpass filtering (5-15 Hz)
  - Derivative and moving average filtering
  - Automatic peak detection with adaptive thresholding
  - Heart rate and RR interval extraction
- **HRV Score**: Comprehensive heart rate variability assessment
  - 0-100 normalized scoring system
  - Trend analysis (improving/declining detection)
  - Stability penalties for high variability
  - Support for sigmoid and percentile scoring methods
  - 30-day baseline comparison

#### Sensor Data Types
- `IMUData`: Base class for inertial measurement unit data with magnitude property
- `AccelerometerData`: Specialized accelerometer sensor handling
- `EcgData`: Electrocardiogram signal container
- `HrvData`: Heart rate variability metrics storage
- Added `to_matrix()` method to convert 3-axis data to (N, 3) numpy arrays

#### DSP Utilities
- `convolution` module with moving average filtering (mov_mean)

#### Documentation
- Comprehensive README with quick start guide
- Detailed algorithm documentation in `/docs/algorithms/`
- MkDocs-based documentation site with Material theme
- Individual algorithm pages covering:
  - Algorithm description and mathematics
  - Parameter explanations
  - Usage examples with code snippets
  - Clinical/research applications
  - References and limitations

#### Testing
- Unit tests for activity algorithms:
  - `test_activity_enmo.py`: ENMO algorithm tests
  - `test_activity_zero_crossing.py`: Zero crossing algorithm tests
  - `test_activity_time_above_thr.py`: Time above threshold tests
  - `test_activity_pim.py`: PIM algorithm tests
  - `test_activity_score.py`: Activity Score tests with randomized baseline data
- Parametrized tests covering multiple sampling frequencies and configurations
- Edge case testing (all above/below threshold scenarios)
- Test data: Example accelerometer CSV file for reproducible tests
- Realistic test data generation with random ranges for robust algorithm testing

#### CI/CD
- GitHub Actions workflow for automated documentation deployment
- GitHub Actions workflow for running unit tests on push/pull requests
  - Tests across Python 3.11 and 3.12
  - Coverage reporting with Codecov integration
  - Dependency caching for faster builds

#### Configuration Files
- `pyproject.toml`: Project metadata and dependencies
- `mkdocs.yaml`: Documentation site configuration
- GitHub workflows for CI/CD

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [Unreleased]

### Planned Features
- Additional activity algorithms (energy expenditure, activity recognition)
- Advanced filtering options (bandpass filters, wavelet analysis)
- Real-time processing capabilities
- Performance optimizations for large datasets
- Extended sensor support (pressure sensors, temperature, humidity)
- Machine learning-based activity classification
