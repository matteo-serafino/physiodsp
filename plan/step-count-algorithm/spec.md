# Specification: Step Count Algorithm

## 1. Requirements

### Functional Requirements
- **Target Device**: Wrist-worn accelerometer.
- **Speed Range**: Accurate step counting and cadence estimation from 2.4 km/h (slow walk) to 15 km/h (running).
- **False Positive Suppression**: 
    - High rejection rate for sedentary/sleep periods.
    - Strong differentiation between intentional locomotion and Activities of Daily Living (ADL) such as gesturing, cooking, or tooth-brushing.
- **Temporal Resolution**: Near-real-time processing using a window-based approach.

### Performance Targets
- **Accuracy (MAPE)**: Mean Absolute Percentage Error < 10% vs ground truth step counts.
- **Cadence Accuracy**: Cadence MAPE < 5% during steady-state gait.
- **Robustness**: F1-score > 0.85 across walking and running classes; precision > 0.90 during sleep.

### Technical Constraints
- **Sampling Rate**: Target 50 Hz to balance temporal resolution (Nyquist for 3.5 Hz cadence) and power consumption.
- **Orientation Independence**: Must handle arbitrary wrist rotation.

---

## 2. Design Rationales

The algorithm employs a five-stage pipeline to progressively filter noise and isolate gait signals:

| Stage | Component | Rationale |
| :--- | :--- | :--- |
| **1. Pre-processing** | Vector Magnitude (VM) | Ensures rotation invariance; captures movement regardless of wrist orientation. |
| | HP Butterworth (0.5 Hz) | Removes gravity (1g DC) and slow drifts while preserving walking dynamics ($\ge 1.2$ Hz). |
| **2. Motion Gate** | SMA Threshold | Early exit for sleep/rest windows ($\text{SMA} < 0.02\text{g}$) to minimize computational cost and false positives. |
| **3. Periodicity** | Autocorrelation | Locomotion is uniquely periodic. ADL gestures are typically aperiodic or lack the sustained rhythm of gait. |
| | PSD Spectral Purity | Secondary confirmation for ambiguous windows by checking for a dominant harmonic peak. |
| **4. Detection** | Adaptive Band-Pass | Centers the filter around the detected cadence to reject harmonics and environmental noise. |
| | Adaptive RMS Threshold | Tracks local signal amplitude to maintain sensitivity from slow walks ($\sim 0.3\text{g}$) to sprints ($\sim 3\text{g}$). |
| **5. Validation** | Bout Continuity | Requires a minimum of 4 consecutive steps to credit a bout, eliminating isolated gesture-triggered steps. |
| | Interpolation | Fills brief gaps caused by sensor saturation or occlusion using the prevailing cadence. |

---

## 3. Implementation Details & Examples

### Core Pipeline Logic
The signal is processed through the following sequence:
`3-axis Accel` $\rightarrow$ `VM` $\rightarrow$ `HP Filter` $\rightarrow$ `SMA Gate` $\rightarrow$ `Autocorrelation` $\rightarrow$ `Adaptive BP Filter` $\rightarrow$ `Adaptive Peak Detection` $\rightarrow$ `Bout Validation`.

### Key Algorithmic Snippets

#### Periodicity Detection (Autocorrelation)
```python
# Normalized autocorrelation to find dominant gait frequency
norm = np.correlate(sig, sig, mode='full')[len(norm)//2:]
norm /= norm[0]
# Search for peak in the 0.5 - 3.5 Hz range
peak_val = np.max(norm[lag_min:lag_max])
is_periodic = peak_val > 0.5 # Threshold for gait confirmation
```

#### Adaptive Peak Thresholding
```python
# Threshold is a fraction of the local RMS amplitude
local_rms = np.sqrt(np.convolve(signal**2, np.ones(window)/window, mode='same'))
threshold = 0.3 * local_rms # Adaptive scaling
```

### Usage Example
```python
from physiodsp.activity.step_count import StepCount, StepCountSettings
from physiodsp.sensors.imu.accelerometer import AccelerometerData

# Initialize algorithm with custom bout gap
sc = StepCount(settings=StepCountSettings(bout_merge_gap_s=5.0))
result = sc.run(accel_data)

print(f"Total steps: {result.total_steps}")
print(result.biomarker.head()) # DataFrame with timestamps, cadence, and activity state
```

---

## 4. References & Validation

### Validation Datasets
- **OxWalk**: Primary source for free-living wrist acceleration and video ground truth.
- **Clemson Ped-Eval**: Used for validating the full speed range on treadmills.
- **MAREA**: Used for auditing ADL false positives in free-living conditions.

### Technical References
- **Classical DSP Approach**: The current implementation follows a deterministic pipeline for explainability and low-power deployment.
- **ML Baseline**: The "stepcount" (OxWalk) self-supervised ML approach serves as the state-of-the-art benchmark for future upgrades (Target F1 $\approx 0.89$).
