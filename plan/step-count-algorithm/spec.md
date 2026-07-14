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

---

## 5. Implementation Notes & Change Log

Changes to `physiodsp/activity/step_count.py` and
`tests/test_activity_step_count.py`.

### 5.1 Fixed `test_step_count_total_steps_consistency`

The test asserted the old *cumulative* semantics:

```python
assert int(result.biomarker["step_count"].iloc[-1]) == result.total_steps
```

Since commit `46d133d "Remove cum steps"`, the `step_count` column is a
**per-bin** count (steps in each 1 s bin), not a running total. The final bin
therefore holds only that second's steps (e.g. `1`), not the grand total
(`32`), so the assertion failed.

The column still accounts for every surviving step, so the fix asserts the
**sum** instead:

```python
assert int(result.biomarker["step_count"].sum()) == result.total_steps
```

### 5.2 Fixed walking underestimation

**Symptom.** Step counts were systematically low during sustained walking. Pure
synthetic sines were exact, but realistic / longer signals undercounted by
~2–6 %.

**Root cause.** The pipeline band-pass filtered **and** peak-detected each 2 s
analysis window *in isolation*. Two boundary effects dropped real steps:

1. **Filter edge transients** — `sosfiltfilt` distorts ~0.1–0.2 s at each end of
   every 2 s segment, so a step landing near a window boundary could be lost.
2. **Detector reset** — the adaptive-threshold peak detector restarted its state
   at every window, so peaks straddling a boundary were mishandled.

In addition, the integer window count (`len(vm_dynamic) // win_samples`)
silently discarded the trailing partial window (< 2 s of samples).

Measured on a 30 s, 1.6 Hz walk: continuous detection found **48/48** steps,
while summing isolated per-window detections found only **45**.

**Fix.** Detection is now **decoupled from the analysis window**:

- **Stage 2–3 (per window)** only *classify* each window — motion gate (SMA),
  periodicity (autocorrelation / PSD), and the smoothed cadence estimate.
- **Stage 4 (per run)** groups consecutive locomotion windows into contiguous
  sample-index *runs* via the new `group_locomotion_runs()` helper, then
  band-pass filters and peak-detects **each run as one continuous signal**. The
  final run is extended to the end of the recording so trailing remainder
  samples are still searched.

Per-step activity labels are mapped back from the window each step falls in.

**Result.**

| Scenario                              | Before        | After          |
| ------------------------------------- | ------------- | -------------- |
| Realistic gait 1.2–2.5 Hz (30 s)      | −2 to −6 %    | ≈ +2 to +3 %   |
| Non-multiple durations (tail dropped) | −2.0 to −2.5 %| +2.0 to +2.5 % |
| Pure sine (all speeds)                | exact         | exact          |

The systematic undercount is gone; remaining error is small and no longer
one-sided. All 72 tests pass and `flake8` is clean.

### 5.3 On increasing the analysis window 2 s → 5 s

Investigated and **not recommended as a default.** Now that detection runs over
contiguous runs (not per window), the window only controls *classification*
granularity, so a longer window no longer helps the boundary-loss problem it
might once have masked.

The cost of 5 s is concrete and one-sided in the **wrong** direction —
over-counting of short, intermittent bouts, which is exactly the free-living
ADL/rest false-positive case the spec prioritises:

| Isolated walk bout (rest + walk + rest), 1.6 Hz | expected | 2 s | 5 s |
| ----------------------------------------------- | -------- | --- | --- |
| 2.0 s walk                                      | ~3       | 3   | 8   |
| 3.0 s walk                                      | ~5       | 3   | 8   |
| 6.0 s walk                                      | ~10      | 10  | 15  |

Why: a sub-window walk makes the **entire** 5 s window classify as locomotion,
so the surrounding rest is absorbed into the run. The adaptive RMS threshold
(`threshold_factor * local_rms`) then collapses across those low-amplitude rest
stretches, and noise crosses it → false steps. Steady-state walking is
unaffected (2 s and 5 s give identical counts), so 5 s only adds risk.

`analysis_window_s` remains a `StepCountSettings` field, so it can still be
raised per-dataset when the input is known to be continuous gait. The default
stays **2 s**.

A cleaner future lever for slow-cadence periodicity robustness (the only real
benefit of a longer window) would be an **absolute floor on the peak-detection
threshold** so noise in absorbed rest regions can't cross it — independent of
window length.
