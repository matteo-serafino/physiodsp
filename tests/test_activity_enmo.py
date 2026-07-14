import os
import pytest
from datetime import datetime, timezone

from numpy import arange
from pandas import read_csv
from physiodsp.sensors.imu.accelerometer import AccelerometerData
from physiodsp.activity.enmo import ENMO


test_folder_path = os.path.dirname(os.path.realpath(__file__))

# Fixed, minute-aligned reference timestamp. ENMO.aggregate() bins timestamps
# by flooring to the aggregation window (e.g. `// 60 * 60`), so seeding from
# datetime.now() made the test flaky: whenever it happened to run within a
# few seconds of a real minute boundary, the generated samples spanned two
# bins instead of one.
FIXED_TIMESTAMP_START = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize(
        "n_samples,fs",
        [
            (128, 32),
            (256, 64),
            (256, 32)
        ]
)
def test_activity_enmo(n_samples, fs):

    df = read_csv(os.path.join(test_folder_path, "accelerometer.csv"), usecols=["x", "y", "z"])

    timestamps = FIXED_TIMESTAMP_START + arange(start=0, step=1/fs, stop=int(n_samples/fs))

    accelerometer = AccelerometerData(
        timestamps=timestamps,
        x=df.x.values[:n_samples],
        y=df.y.values[:n_samples],
        z=df.z.values[:n_samples],
        fs=fs
    )

    enmo_processor = ENMO().run(accelerometer=accelerometer).aggregate()

    enmo_1s = enmo_processor.biomarker
    enmo_60s = enmo_processor.biomarker_agg

    assert len(enmo_1s) == int(n_samples / fs) - 1
    assert len(enmo_60s) == 1
