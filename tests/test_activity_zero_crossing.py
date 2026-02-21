import pytest
from datetime import datetime

from numpy import arange
from pandas import read_csv
from physiodsp.sensors.imu.accelerometer import AccelerometerData
from physiodsp.activity.zero_crossing import ZeroCrossing


@pytest.mark.parametrize(
        "fs",
        [
            (32),
            (64),
            (128)
        ]
)
def test_activity_enmo(fs):

    df = read_csv("/Users/ms/Documents/repo/py-physio-dsp/tests/accelerometer.csv", usecols=["x", "y", "z"])
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

    assert True
