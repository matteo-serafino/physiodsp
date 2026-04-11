import os
import pytest
from datetime import datetime

from numpy import arange, zeros
from pandas import read_csv
from physiodsp.sensors.imu.accelerometer import AccelerometerData
from physiodsp.balance_tests.sway import Sway

test_folder_path = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
        "fs",
        [
            (25),
        ]
)
def test_balance_test_sway(fs):

    df = read_csv(os.path.join(test_folder_path, "sway.csv"), usecols=["x", "z"])
    n_samples = len(df)

    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))

    accelerometer = AccelerometerData(
        timestamps=timestamps,
        x=df.x.values[:len(timestamps)],
        y=zeros(len(timestamps)),
        z=df.z.values[:len(timestamps)],
        fs=fs
    )

    processor = Sway().run(accelerometer=accelerometer, sensor_height=1.4)

    metrics = processor.biomarker

    assert metrics is not None
    assert hasattr(metrics, "shape")
    assert hasattr(metrics, "columns")
    assert hasattr(metrics, "empty")
    assert not metrics.empty
    assert metrics.shape[0] == 3
