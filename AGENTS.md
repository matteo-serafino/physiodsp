# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setting up the Development Environment
```bash
# Install base dependencies
pip install -r requirements.txt

# Install development dependencies  
pip install -r requirements-dev.txt

# Install test dependencies
pip install -r requirements-test.txt
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run tests with coverage (as used in CI)
coverage run -m pytest

# Run a specific test file
pytest tests/test_activity_enmo.py

# Run a specific test function
pytest tests/test_activity_enmo.py::test_activity_enmo
```

### Code Quality
```bash
# Run flake8 linting
flake8 physiodsp/ tests/

# Auto-format is not configured; maintain existing code style
```

## Project Structure

### Core Architecture
PhysioDSP follows a consistent algorithm architecture pattern:

1. **Settings Models**: Pydantic BaseModel subclasses defining algorithm parameters
2. **Algorithm Classes**: Inherit from `BaseAlgorithm` with standardized `run()` method
3. **Data Models**: Type-safe sensor data classes in `physiodsp.sensors.*` modules
4. **Results**: Biomarkers stored in Pandas DataFrames accessible via `.biomarker` (raw) and `.biomarker_agg` (aggregated) attributes

### Main Modules
- `activity/`: Activity analysis algorithms (ENMO, Zero Crossing, Time Above Threshold, PIM, Activity Score)
- `sensors/`: Sensor data classes (AccelerometerData, EcgData, HrvData, IMU components)
- `ecg/`: ECG signal processing (QRS peak detection)
- `hrv/`: Heart rate variability analysis
- `balance_tests/`: Postural sway analysis
- `dsp/`: Digital signal processing utilities
- `base.py`: Abstract `BaseAlgorithm` class defining the common interface

### Key Conventions
- All algorithms inherit from `BaseAlgorithm` in `physiodsp/base.py`
- Settings are defined using Pydantic models with field validation
- Sensor data classes handle validation and unit consistency
- Algorithm results include both raw biomarker data and aggregated values
- Version information is stored in `_version.py` and accessed via class properties
- Test files follow naming pattern `test_<module>_<algorithm>.py`

### Data Flow
1. Create sensor data object (e.g., `AccelerometerData`)
2. Initialize algorithm with settings (e.g., `ENMO(settings=ENMOSettings())`)
3. Run algorithm: `result = algorithm.run(sensor_data)`
4. Access results: `result.biomarker` (DataFrame with timestamps/values)
5. Optionally aggregate: `result.aggregate(method='mean')` then access `result.biomarker_agg`

## Contributing Guidelines

When adding new functionality:
- Follow the existing algorithm architecture pattern
- Inherit from `BaseAlgorithm` and implement `run()` method
- Define settings using Pydantic BaseModel
- Include comprehensive unit tests in the `tests/` directory
- Maintain type hints throughout
- Update README.md documentation with usage examples
- Ensure flake8 passes with current configuration (max line length 120, ignore E501, E722)

## CI/CD Pipeline
GitHub Actions workflows automatically:
- Run tests on Python 3.11 and 3.12 for pushes to main/develop branches
- Run tests on pull requests targeting main/develop branches
- Generate coverage reports

## Versioning
Version is dynamically managed via `physiodsp._version.__version__` and defined in pyproject.toml using setuptools dynamic versioning.

## Git contract

- Commit messages follow **Conventional Commits**.
- Signed commits: yes.
- Co-Authored-By trailer: no.
- **Integration:** direct-to-main, **fast-forward only**. No merge
  commits on `main`.

## Documentation contract
* Update the README.md when major changes have been applied (e.g., addition of a new algorithm)
* Update the documentation in docs folder
