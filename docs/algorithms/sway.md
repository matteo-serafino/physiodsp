# Sway Analysis

## Overview

The Sway algorithm quantifies postural sway from accelerometer data during balance tests. It computes center-of-pressure-proxy displacement signals in the medio-lateral (ML) and antero-posterior (AP) directions and extracts a set of stabilometric indices and a 95% confidence ellipse.

**Algorithm Name:** Sway  
**Version:** 0.1.0

## Algorithm Description

1. **Displacement estimation**: ML and AP displacements (in meters) are approximated by multiplying the accelerometer signal by the sensor height using a small-angle approximation. The accelerometer axes must be expressed in **g** (gravitational units, where 1 g ≈ 9.81 m/s²); under a small-angle tilt, the horizontal component equals approximately the tilt angle in radians, so that:
$$d_{ML} = x \cdot h, \quad d_{AP} = z \cdot h$$
where $x$ and $z$ are in g and $h$ is the sensor height in meters. If raw sensor values are in m/s², divide by 9.81 before passing them to the algorithm.
2. **Low-pass filtering**: A zero-phase Butterworth filter removes high-frequency noise above the cutoff frequency (default 2.5 Hz).
3. **Mean removal**: The mean of each filtered signal is subtracted to center the trajectory around zero.
4. **Sway index extraction**: Path-based stabilometric indices are computed for the full resultant path, the ML path, and the AP path.
5. **Confidence ellipse**: A 95% confidence ellipse is fitted to the ML–AP scatter using eigenvalue decomposition of the covariance matrix; for this symmetric covariance matrix, this is equivalent to using SVD.

## Parameters

### SwaySettings

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter_order` | `PositiveInt` | `4` | Butterworth filter order for low-pass filtering |
| `filter_high_freq` | `PositiveFloat` | `2.5` | Low-pass cutoff frequency in Hz |

## Usage Example

```python
from physiodsp.balance_tests.sway import Sway, SwaySettings
from physiodsp.sensors.imu.accelerometer import AccelerometerData
import numpy as np

# Create sample accelerometer data (50 Hz, 30 seconds)
fs = 50
duration = 30
timestamps = np.arange(0, duration, 1 / fs)
n = len(timestamps)

accel_data = AccelerometerData(
    timestamps=timestamps,
    x=np.random.normal(0, 0.01, n),   # ML axis (g)
    y=np.random.normal(1.0, 0.01, n), # vertical axis (g, ≈ 1 g at rest)
    z=np.random.normal(0, 0.01, n),   # AP axis (g)
    fs=fs
)

# Initialize with custom settings
settings = SwaySettings(filter_order=4, filter_high_freq=2.5)
sway = Sway(settings=settings)

# Run algorithm (sensor height in meters)
result = sway.run(accel_data, sensor_height=1.0)

# Access results
print(result.biomarker)
```

## Output

The algorithm returns a Pandas DataFrame (`biomarker`) with the following columns:

### Metadata columns

| Column | Description |
|---|---|
| `index` | Path descriptor: `full_path`, `ml_path`, or `ap_path` |
| `timestamp_start` | Start timestamp of the recording |
| `timestamp_end` | End timestamp of the recording |
| `duration_s` | Duration of the recording in seconds |

### Sway index columns

| Column | Unit | Description |
|---|---|---|
| `average_distance_m` | m | Mean displacement from the center |
| `rms_distance_m` | m | Root mean square displacement |
| `total_distance_m` | m | Total path length traveled |
| `average_velocity_m_s` | m/s | Mean sway velocity (total distance / duration) |

### Ellipse columns (`full_path` row only)

These columns are populated **only for the `full_path` row**. The `ml_path` and `ap_path` rows contain `NaN` for all ellipse columns, because the confidence ellipse requires both ML and AP signals jointly.

| Column | Unit | Description |
|---|---|---|
| `ellipse_area_m2` | m² | Area of the 95% confidence ellipse ($\pi \cdot \chi^2 \cdot \sqrt{\lambda_1 \lambda_2}$, $\chi^2 = 4.605$) |
| `theta_rad` | rad | Tilt angle of the ellipse major axis |
| `a_m` | m | Semi-major axis length ($\sqrt{\lambda_1 \cdot \chi^2}$) |
| `b_m` | m | Semi-minor axis length ($\sqrt{\lambda_2 \cdot \chi^2}$) |

## Signal Processing Details

### Low-pass Filtering

A zero-phase (forward–backward) Butterworth filter is applied to remove high-frequency noise while preserving the low-frequency sway motion:

$$|H(j\omega)|^2 = \frac{1}{1 + (\omega / \omega_c)^{2N}}$$

where $N$ is the filter order, $\omega$ is the angular frequency, and $\omega_c = 2\pi \cdot f_c$ is the cutoff angular frequency.

### 95% Confidence Ellipse

The ellipse parameters are derived from singular value decomposition (SVD) of the ML–AP covariance matrix $\Sigma$:

$$\Sigma = U \, S \, V^\top$$

- **Area**: $\pi \cdot \chi^2_{0.95} \cdot \sqrt{s_1 \cdot s_2}$  
- **Axes**: $a = \sqrt{s_1 \cdot \chi^2_{0.95}}$, $b = \sqrt{s_2 \cdot \chi^2_{0.95}}$  
- **Angle**: $\theta = \arctan(U_{10} / U_{00})$

The $\chi^2$ threshold for a 2-DOF 95% confidence interval is $4.605$.

## Clinical / Research Applications

- Clinical balance assessment (e.g., Romberg test, single-leg stance)
- Fall risk screening in elderly populations
- Rehabilitation outcome monitoring
- Sports science and performance analysis

## References

- Prieto, T. E. et al. (1996). Measures of postural steadiness: differences between healthy young and elderly adults. *IEEE Transactions on Biomedical Engineering*, 43(9), 956–966.
- Chiari, L. et al. (2000). Stabilometric parameters are affected by anthropometry and foot placement. *Clinical Biomechanics*, 15(9), 666–678.
