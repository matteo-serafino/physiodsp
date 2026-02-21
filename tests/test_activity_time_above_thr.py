import pytest
from datetime import datetime

from numpy import arange, ones, zeros
from pandas import read_csv
from physiodsp.sensors.imu.base import IMUData
from physiodsp.activity.time_above_thr import TimeAboveThr, TimeAboveThrSettings


@pytest.mark.parametrize(
        "n_samples,fs,threshold",
        [
            (128, 32, 0.1),
            (256, 64, 0.2),
            (256, 32, 0.05),
            (512, 100, 0.15)
        ]
)
def test_activity_time_above_thr(n_samples, fs, threshold):
    """Test TimeAboveThr algorithm with various parameters"""

    df = read_csv("/Users/ms/Documents/repo/py-physio-dsp/tests/accelerometer.csv", usecols=["x", "y", "z"])

    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))

    imu_data = IMUData(
        timestamps=timestamps,
        x=df.x.values[:n_samples],
        y=df.y.values[:n_samples],
        z=df.z.values[:n_samples],
        fs=fs
    )

    tat_processor = TimeAboveThr(settings=TimeAboveThrSettings(threshold=threshold)).run(imu_data)
    tat_1s = tat_processor.values

    # Check output structure
    assert "timestamps" in tat_1s.columns
    assert "x" in tat_1s.columns
    assert "y" in tat_1s.columns
    assert "z" in tat_1s.columns

    # Check output length (rolling window with 1s step)
    assert len(tat_1s) > 0
    assert len(tat_1s) == int(n_samples / fs) - 1

    # Values should be counts (non-negative integers)
    assert (tat_1s["x"] >= 0).all()
    assert (tat_1s["y"] >= 0).all()
    assert (tat_1s["z"] >= 0).all()


def test_time_above_thr_all_above_threshold():
    """Test when all samples are above threshold"""

    n_samples = 128
    fs = 32
    threshold = 0.01  # Very low threshold

    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))

    # Create data with values well above threshold
    x_data = ones(n_samples) * 1.0
    y_data = ones(n_samples) * 1.0
    z_data = ones(n_samples) * 1.0

    imu_data = IMUData(
        timestamps=timestamps,
        x=x_data,
        y=y_data,
        z=z_data,
        fs=fs
    )

    tat_processor = TimeAboveThr(settings=TimeAboveThrSettings(threshold=threshold)).run(imu_data)
    tat_1s = tat_processor.values

    # All samples should be above threshold
    # With 1s windows at 32 Hz, each window should have 32 samples above threshold
    assert (tat_1s["x"] == fs).all()
    assert (tat_1s["y"] == fs).all()
    assert (tat_1s["z"] == fs).all()


def test_time_above_thr_all_below_threshold():
    """Test when all samples are below threshold"""

    n_samples = 128
    fs = 32
    threshold = 10.0  # Very high threshold

    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))

    # Create data with values well below threshold
    x_data = zeros(n_samples)
    y_data = zeros(n_samples)
    z_data = zeros(n_samples)

    imu_data = IMUData(
        timestamps=timestamps,
        x=x_data,
        y=y_data,
        z=z_data,
        fs=fs
    )

    tat_processor = TimeAboveThr(settings=TimeAboveThrSettings(threshold=threshold)).run(imu_data)
    tat_1s = tat_processor.values

    # No samples should be above threshold
    assert (tat_1s["x"] == 0).all()
    assert (tat_1s["y"] == 0).all()
    assert (tat_1s["z"] == 0).all()


def test_time_above_thr_custom_settings():
    """Test TimeAboveThr with custom settings"""

    n_samples = 256
    fs = 64
    window_len = 2
    aggregation_window = 120
    threshold = 0.15

    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))

    df = read_csv("/Users/ms/Documents/repo/py-physio-dsp/tests/accelerometer.csv", usecols=["x", "y", "z"])

    imu_data = IMUData(
        timestamps=timestamps,
        x=df.x.values[:n_samples],
        y=df.y.values[:n_samples],
        z=df.z.values[:n_samples],
        fs=fs
    )

    settings = TimeAboveThrSettings(
        window_len=window_len,
        aggregation_window=aggregation_window,
        threshold=threshold
    )

    tat_processor = TimeAboveThr(settings=settings).run(imu_data)
    tat_results = tat_processor.values

    # Check that output is generated
    assert len(tat_results) > 0

    # With 2s window at 64 Hz, each window should have 128 samples
    # Number of windows should be approximately n_samples / (window_len * fs)
    expected_windows = int(n_samples / (window_len * fs)) - 1
    assert len(tat_results) == expected_windows

    # All columns should be present
    assert "timestamps" in tat_results.columns
    assert "x" in tat_results.columns
    assert "y" in tat_results.columns
    assert "z" in tat_results.columns


def test_time_above_thr_algorithm_properties():
    """Test basic properties of TimeAboveThr algorithm"""

    tat = TimeAboveThr()

    # Check algorithm metadata
    assert tat.algorithm_name == "TimeAboveThrAlgorithm"
    assert tat.version == "v0.1.0"

    # Check default settings
    default_settings = TimeAboveThrSettings()
    assert default_settings.window_len == 1
    assert default_settings.aggregation_window == 60
    assert default_settings.threshold == 0.1


def test_time_above_thr_output_structure():
    """Test output DataFrame structure"""

    n_samples = 128
    fs = 32

    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))

    df = read_csv("/Users/ms/Documents/repo/py-physio-dsp/tests/accelerometer.csv", usecols=["x", "y", "z"])

    imu_data = IMUData(
        timestamps=timestamps,
        x=df.x.values[:n_samples],
        y=df.y.values[:n_samples],
        z=df.z.values[:n_samples],
        fs=fs
    )

    tat_processor = TimeAboveThr().run(imu_data)

    # Check that values attribute exists
    assert hasattr(tat_processor, 'values')
    assert tat_processor.values is not None

    # Check column data types
    assert tat_processor.values["timestamps"].dtype in ['float64', 'float32']
    assert tat_processor.values["x"].dtype in ['int64', 'int32', 'float64', 'float32']
    assert tat_processor.values["y"].dtype in ['int64', 'int32', 'float64', 'float32']
    assert tat_processor.values["z"].dtype in ['int64', 'int32', 'float64', 'float32']
