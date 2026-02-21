
import os
import pytest
from datetime import datetime
from numpy import arange, ones, zeros
from pandas import read_csv
from physiodsp.sensors.imu.base import IMUData
from physiodsp.activity.pim import PIMAlgorithm


test_folder_path = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
    "n_samples,fs",
    [
        (128, 32),
        (256, 64),
        (512, 100)
    ]
)
def test_pim_algorithm(n_samples, fs):
    """Test PIM algorithm with various parameters"""
    df = read_csv(os.path.join(test_folder_path, "accelerometer.csv"), usecols=["x", "y", "z"])
    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))
    imu_data = IMUData(
        timestamps=timestamps,
        x=df.x.values[:n_samples],
        y=df.y.values[:n_samples],
        z=df.z.values[:n_samples],
        fs=fs
    )
    pim = PIMAlgorithm()
    result = pim.estimate(imu_data)
    assert hasattr(result, 'values_x')
    assert hasattr(result, 'values_y')
    assert hasattr(result, 'values_z')
    assert len(result.values_x) == n_samples
    assert len(result.values_y) == n_samples
    assert len(result.values_z) == n_samples
    assert (result.values_x >= 0).all()
    assert (result.values_y >= 0).all()
    assert (result.values_z >= 0).all()


def test_pim_absolute_values():
    """Test that PIM correctly computes absolute values"""
    n_samples = 64
    fs = 32
    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))
    x_data = arange(-10, 10, 20/n_samples)
    y_data = arange(-5, 5, 10/n_samples)
    z_data = arange(-2, 2, 4/n_samples)
    imu_data = IMUData(
        timestamps=timestamps,
        x=x_data,
        y=y_data,
        z=z_data,
        fs=fs
    )
    pim = PIMAlgorithm()
    result = pim.estimate(imu_data)
    import numpy as np
    assert np.allclose(result.values_x, np.abs(x_data))
    assert np.allclose(result.values_y, np.abs(y_data))
    assert np.allclose(result.values_z, np.abs(z_data))


def test_pim_positive_values():
    """Test PIM with already positive values"""
    n_samples = 128
    fs = 32
    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))
    x_data = ones(n_samples) * 0.5
    y_data = ones(n_samples) * 0.3
    z_data = ones(n_samples) * 0.7
    imu_data = IMUData(
        timestamps=timestamps,
        x=x_data,
        y=y_data,
        z=z_data,
        fs=fs
    )
    pim = PIMAlgorithm()
    result = pim.estimate(imu_data)
    import numpy as np
    assert np.allclose(result.values_x, x_data)
    assert np.allclose(result.values_y, y_data)
    assert np.allclose(result.values_z, z_data)


def test_pim_zero_values():
    """Test PIM with zero values"""
    n_samples = 64
    fs = 32
    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))
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
    pim = PIMAlgorithm()
    result = pim.estimate(imu_data)
    assert (result.values_x == 0).all()
    assert (result.values_y == 0).all()
    assert (result.values_z == 0).all()


def test_pim_algorithm_properties():
    """Test basic properties of PIM algorithm"""
    pim = PIMAlgorithm()
    assert pim.algorithm_name == "PIMAlgorithm"
    assert pim.version == "v0.1.0"
    assert hasattr(pim, '_aggregation_window')
    assert pim._aggregation_window == 5


def test_pim_estimate_returns_self():
    """Test that estimate method returns self for method chaining"""
    df = read_csv(os.path.join(test_folder_path, "accelerometer.csv"), usecols=["x", "y", "z"])
    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/32, stop=4)
    imu_data = IMUData(
        timestamps=timestamps,
        x=df.x.values[:128],
        y=df.y.values[:128],
        z=df.z.values[:128],
        fs=32
    )
    pim = PIMAlgorithm()
    result = pim.estimate(imu_data)
    assert result is pim


def test_pim_data_preservation():
    """Test that original IMU data is not modified"""
    n_samples = 64
    fs = 32
    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))
    x_data = arange(-10, 10, 20/n_samples)
    y_data = arange(-5, 5, 10/n_samples)
    z_data = arange(-2, 2, 4/n_samples)
    x_copy = x_data.copy()
    y_copy = y_data.copy()
    z_copy = z_data.copy()
    imu_data = IMUData(
        timestamps=timestamps,
        x=x_data,
        y=y_data,
        z=z_data,
        fs=fs
    )
    pim = PIMAlgorithm()
    pim.estimate(imu_data)
    import numpy as np
    assert np.allclose(imu_data.x, x_copy)
    assert np.allclose(imu_data.y, y_copy)
    assert np.allclose(imu_data.z, z_copy)


def test_pim_mixed_values():
    """Test PIM with mixed positive and negative values"""
    n_samples = 100
    fs = 50
    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))
    x_data = arange(-5, 5, 10/n_samples)
    y_data = arange(0, 10, 10/n_samples)
    z_data = arange(-10, 0, 10/n_samples)
    imu_data = IMUData(
        timestamps=timestamps,
        x=x_data,
        y=y_data,
        z=z_data,
        fs=fs
    )
    pim = PIMAlgorithm()
    result = pim.estimate(imu_data)
    import numpy as np
    assert np.allclose(result.values_x, np.abs(x_data))
    assert np.allclose(result.values_y, np.abs(y_data))
    assert np.allclose(result.values_z, np.abs(z_data))
    assert (result.values_x >= 0).all()
    assert (result.values_y >= 0).all()
    assert (result.values_z >= 0).all()
