import os
import pytest
from datetime import datetime

from numpy import arange
from pandas import read_csv
from physiodsp.sensors.imu.accelerometer import AccelerometerData
from physiodsp.activity.zero_crossing import ZeroCrossing

test_folder_path = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
        "fs",
        [
            (32),
            (64),
            (128)
        ]
)
def test_activity_zero_crossing(fs):

    df = read_csv(os.path.join(test_folder_path, "accelerometer.csv"), usecols=["x", "y", "z"])
    n_samples = len(df)

    timestamp_start = datetime.now().timestamp()
    timestamps = timestamp_start + arange(start=0, step=1/fs, stop=int(n_samples/fs))

    accelerometer = AccelerometerData(
        timestamps=timestamps,
        x=df.x.values[:len(timestamps)],
        y=df.y.values[:len(timestamps)],
        z=df.z.values[:len(timestamps)],
        fs=fs
    )

    processor = ZeroCrossing().run(data=accelerometer).aggregate()

    zcr_1s = processor.biomarker
    zcr_60s = processor.biomarker_agg

    assert len(zcr_1s) == int(n_samples / fs) - 1
    assert len(zcr_60s) >= 1
    assert "x" in zcr_1s.columns
    assert "y" in zcr_1s.columns
    assert "z" in zcr_1s.columns
    assert "timestamps" in zcr_1s.columns
    assert zcr_1s["x"].dtype in [int, float]
    assert zcr_1s["y"].dtype in [int, float]
    assert zcr_1s["z"].dtype in [int, float]
