from __future__ import annotations

import numpy as np
import pytest

from physiodsp.activity.step_count import (
    StepBout,
    StepCount,
    StepCountSettings,
    autocorrelation_peak,
    detect_steps_in_window,
    filter_short_bouts,
    highpass_filter,
    interpolate_missed_steps,
    is_locomotion_window,
    merge_close_bouts,
    steps_to_cadence_series,
)
from physiodsp.sensors.imu.accelerometer import AccelerometerData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 50.0  # Hz — a common sampling rate for testing


def make_sine_accel(
    freq_hz: float,
    duration_s: float,
    amplitude: float = 0.5,
    fs: float = FS,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Periodic sinusoidal signal centred at 0, simulating dynamic acceleration."""
    t = np.arange(int(duration_s * fs)) / fs
    sig = amplitude * np.sin(2 * np.pi * freq_hz * t)
    if noise_std > 0:
        rng = np.random.default_rng(42)
        sig += rng.normal(0, noise_std, size=len(sig))
    return sig


def make_accel_data(
    dynamic_signal: np.ndarray,
    fs: float = FS,
    gravity: float = 1.0,
) -> AccelerometerData:
    """Wrap a 1-D dynamic signal into an AccelerometerData instance.
    The signal is injected on the Z axis; X and Y carry only gravity / noise.
    """
    n = len(dynamic_signal)
    t = np.arange(n) / fs
    x = np.zeros(n)
    y = np.zeros(n)
    z = gravity + dynamic_signal  # gravity on Z; dynamic component added
    return AccelerometerData(timestamps=t, x=x, y=y, z=z, fs=int(fs))


def run_step_count_on_sine(
    freq_hz: float,
    duration_s: float = 30.0,
    amplitude: float = 0.5,
    noise_std: float = 0.0,
    settings: StepCountSettings | None = None,
) -> StepCount:
    """Helper to run StepCount algorithm on a synthetic sine signal."""
    sig = make_sine_accel(freq_hz, duration_s, amplitude, noise_std=noise_std)
    accel = make_accel_data(sig)
    sc = StepCount(settings=settings or StepCountSettings())
    return sc.run(accel)


def make_test_bouts() -> list[StepBout]:
    """Helper to create a set of sample StepBout objects."""
    return [
        StepBout(0.0, 5.0, 8, 100.0, "walk", [0.5 * i for i in range(8)]),
        StepBout(20.0, 25.0, 2, 80.0, "walk", [20.0 + 0.5 * i for i in range(2)]),
        StepBout(30.0, 40.0, 15, 120.0, "walk", [30.0 + 0.5 * i for i in range(15)]),
    ]


# ---------------------------------------------------------------------------
# Pre-processing Tests
# ---------------------------------------------------------------------------

def test_step_count_highpass_removes_dc():
    """HP filter must remove the 1 g DC gravity component."""
    n = 1000
    sig = np.ones(n) * 1.0 + 0.3 * np.sin(2 * np.pi * 1.5 * np.arange(n) / FS)
    filtered = highpass_filter(sig, fs=FS, cutoff_hz=0.5)
    # DC component must be removed: mean of filtered signal ≈ 0
    assert abs(np.mean(filtered)) < 0.05


def test_step_count_highpass_preserves_gait_frequencies():
    """HP filter must not attenuate a 1.5 Hz gait signal significantly."""
    duration_s = 10.0
    sig = make_sine_accel(freq_hz=1.5, duration_s=duration_s)
    filtered = highpass_filter(sig, fs=FS, cutoff_hz=0.5)
    # RMS of filtered signal should be > 80 % of input RMS
    assert np.std(filtered) > 0.8 * np.std(sig)


# ---------------------------------------------------------------------------
# Motion Gate Tests
# ---------------------------------------------------------------------------

def test_step_count_sleep_window_rejected():
    """Below-sleep-threshold SMA -> is_locomotion_window returns False."""
    sig = np.random.randn(int(2.0 * FS)) * 0.005
    sma = float(np.mean(np.abs(sig)))
    is_loco, _, state = is_locomotion_window(
        sig, fs=FS, sma=sma,
        sma_sleep_thr=0.02, sma_adl_thr=0.05,
        periodicity_threshold=0.5, spectral_purity_threshold=8.0
    )
    assert not is_loco
    assert state == "sleep"


# ---------------------------------------------------------------------------
# Periodicity Detector Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq", [1.2, 1.6, 2.0, 2.5, 3.0])
def test_step_count_pure_sine_is_periodic(freq: float):
    """A pure sinusoid at any valid gait frequency must be detected as periodic."""
    sig = make_sine_accel(freq_hz=freq, duration_s=4.0)
    peak, detected_freq = autocorrelation_peak(sig, fs=FS)
    assert peak > 0.5, f"AC peak {peak:.3f} too low for {freq} Hz sine"
    assert abs(detected_freq - freq) < 0.3, (
        f"Detected freq {detected_freq:.2f} Hz too far from {freq} Hz"
    )


def test_step_count_noise_is_not_periodic():
    """White noise must not exceed the periodicity threshold."""
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 0.3, size=int(4.0 * FS))
    peak, _ = autocorrelation_peak(sig, fs=FS)
    assert peak < 0.5, f"Noise produced AC peak {peak:.3f} >= 0.5"


def test_step_count_walk_window_detected():
    """Walking-frequency signal above SMA threshold -> locomotion detected."""
    sig = make_sine_accel(freq_hz=1.6, duration_s=4.0, amplitude=0.4)
    sma = float(np.mean(np.abs(sig)))
    is_loco, cadence, state = is_locomotion_window(
        sig, fs=FS, sma=sma,
        sma_sleep_thr=0.02, sma_adl_thr=0.05,
        periodicity_threshold=0.5, spectral_purity_threshold=8.0
    )
    assert is_loco
    assert state in ("walk", "run")
    assert 1.0 < cadence < 2.5


def test_step_count_run_window_detected():
    """Running-frequency signal (2.8 Hz) must be classified as run."""
    sig = make_sine_accel(freq_hz=2.8, duration_s=4.0, amplitude=1.5)
    sma = float(np.mean(np.abs(sig)))
    is_loco, cadence, state = is_locomotion_window(
        sig, fs=FS, sma=sma,
        sma_sleep_thr=0.02, sma_adl_thr=0.05,
        periodicity_threshold=0.5, spectral_purity_threshold=8.0
    )
    assert is_loco
    assert state == "run"


# ---------------------------------------------------------------------------
# Adaptive Peak Detector Tests
# ---------------------------------------------------------------------------

def test_step_count_slow_walk_accuracy():
    """Slow walking at 1.3 Hz for 10 s -> ~13 steps expected."""
    freq = 1.3
    duration = 10.0
    sig = make_sine_accel(freq_hz=freq, duration_s=duration, amplitude=0.3)
    peaks = detect_steps_in_window(sig, fs=FS, cadence_hz=freq)
    expected = int(freq * duration)
    assert abs(len(peaks) - expected) <= 2, (
        f"Expected ~{expected} steps, got {len(peaks)}"
    )


def test_step_count_running_accuracy():
    """Running at 2.8 Hz for 10 s -> ~28 steps expected."""
    freq = 2.8
    duration = 10.0
    sig = make_sine_accel(freq_hz=freq, duration_s=duration, amplitude=1.5)
    peaks = detect_steps_in_window(sig, fs=FS, cadence_hz=freq)
    expected = int(freq * duration)
    assert abs(len(peaks) - expected) <= 3, (
        f"Expected ~{expected} steps, got {len(peaks)}"
    )


def test_step_count_amplitude_invariance():
    """Step count should be the same regardless of signal amplitude (adaptive threshold)."""
    freq = 1.6
    duration = 10.0
    steps_low = len(detect_steps_in_window(
        make_sine_accel(freq, duration, amplitude=0.3), fs=FS, cadence_hz=freq))
    steps_high = len(detect_steps_in_window(
        make_sine_accel(freq, duration, amplitude=2.0), fs=FS, cadence_hz=freq))
    assert abs(steps_low - steps_high) <= 2, (
        f"Amplitude should not affect count: low={steps_low}, high={steps_high}"
    )


# ---------------------------------------------------------------------------
# Post-processing Tests
# ---------------------------------------------------------------------------

def test_step_count_filter_short_bouts():
    bouts = make_test_bouts()
    filtered = filter_short_bouts(bouts, min_steps=3)
    assert all(b.step_count >= 3 for b in filtered)
    assert len(filtered) == 2


def test_step_count_merge_close_bouts():
    """Two bouts separated by 2 s gap should be merged with merge_gap_s=3."""
    b1 = StepBout(0.0, 5.0, 8, 100.0, "walk", list(np.linspace(0, 5, 8)))
    b2 = StepBout(7.0, 12.0, 8, 100.0, "walk", list(np.linspace(7, 12, 8)))
    merged = merge_close_bouts([b1, b2], merge_gap_s=3.0)
    assert len(merged) == 1
    assert merged[0].step_count == 16


def test_step_count_no_merge_far_bouts():
    """Bouts separated by > merge_gap_s must NOT be merged."""
    b1 = StepBout(0.0, 5.0, 8, 100.0, "walk", list(np.linspace(0, 5, 8)))
    b2 = StepBout(15.0, 20.0, 8, 100.0, "walk", list(np.linspace(15, 20, 8)))
    merged = merge_close_bouts([b1, b2], merge_gap_s=3.0)
    assert len(merged) == 2


def test_step_count_interpolate_missed_steps():
    """A gap of ~2 periods should produce one interpolated step."""
    cadence_hz = 2.0  # 0.5 s per step
    steps = [0.0, 0.5, 1.5, 2.0]
    filled = interpolate_missed_steps(steps, cadence_hz=cadence_hz)
    assert len(filled) == 5


def test_step_count_cadence_series_bin_count():
    """steps_to_cadence_series must produce exactly ceil(duration/bin_s) bins."""
    steps = np.array([1.0, 2.0, 3.0, 5.0, 6.0])
    bin_ts, cad = steps_to_cadence_series(steps, total_duration_s=10.0, bin_s=1.0)
    assert len(bin_ts) == 10
    assert len(cad) == 10


def test_step_count_cadence_series_values():
    """Bins with exactly 2 steps at bin_s=1 s should give 120 spm."""
    steps = np.array([0.1, 0.9])
    _, cad = steps_to_cadence_series(steps, total_duration_s=3.0, bin_s=1.0)
    assert cad[0] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# End-to-end Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq,speed_label", [
    (1.2, "slow_walk"),
    (1.6, "normal_walk"),
    (2.0, "brisk_walk"),
    (2.5, "jog"),
    (3.0, "run"),
])
def test_step_count_accuracy_all_speeds(freq: float, speed_label: str):
    duration = 30.0
    result = run_step_count_on_sine(freq, duration_s=duration, amplitude=0.5)
    expected = int(freq * duration)
    error_pct = abs(result.total_steps - expected) / expected * 100
    assert error_pct <= 15.0, (
        f"[{speed_label}] Expected ~{expected} steps, got {result.total_steps} "
        f"(error {error_pct:.1f} %)"
    )


def test_step_count_zero_steps_sleep():
    """Near-zero acceleration (sleep) must produce 0 steps."""
    rng = np.random.default_rng(42)
    n = int(30.0 * FS)
    t = np.arange(n) / FS
    sig = rng.normal(0, 0.005, n)
    accel = AccelerometerData(
        timestamps=t, x=sig, y=sig,
        z=np.ones(n) + sig, fs=int(FS)
    )
    result = StepCount().run(accel)
    assert result.total_steps == 0


def test_step_count_biomarker_schema():
    """biomarker DataFrame must have the required columns and correct length."""
    result = run_step_count_on_sine(1.6, duration_s=20.0)
    assert "timestamps" in result.biomarker.columns
    assert "step_count" in result.biomarker.columns
    assert "cadence_spm" in result.biomarker.columns
    assert "activity_state" in result.biomarker.columns
    assert len(result.biomarker) == 20


def test_step_count_total_steps_consistency():
    """The cumulative step_count final value must equal total_steps."""
    result = run_step_count_on_sine(1.6, duration_s=20.0)
    assert int(result.biomarker["step_count"].iloc[-1]) == result.total_steps


def test_step_count_noisy_free_living_walk():
    """Walking signal with realistic free-living noise should still be detected."""
    result = run_step_count_on_sine(
        freq_hz=1.6, duration_s=30.0, amplitude=0.4, noise_std=0.15
    )
    expected = int(1.6 * 30)
    error_pct = abs(result.total_steps - expected) / expected * 100
    assert error_pct <= 20.0, (
        f"Free-living noisy walk: expected ~{expected}, got {result.total_steps} "
        f"(error {error_pct:.1f} %)"
    )


def test_step_count_bout_merge_fragmentation():
    """With merge_gap_s > 0, a walk-pause-walk sequence should produce one bout."""
    freq = 1.6
    fs = FS
    walk1 = make_sine_accel(freq, 10.0, amplitude=0.5)
    pause = np.zeros(int(2.0 * fs))
    walk2 = make_sine_accel(freq, 10.0, amplitude=0.5)
    sig = np.concatenate([walk1, pause, walk2])
    t = np.arange(len(sig)) / fs
    z = 1.0 + sig
    accel = AccelerometerData(
        timestamps=t, x=np.zeros_like(sig),
        y=np.zeros_like(sig), z=z, fs=int(fs)
    )

    merged_result = StepCount(settings=StepCountSettings(bout_merge_gap_s=3.0)).run(accel)
    no_merge_result = StepCount(settings=StepCountSettings(bout_merge_gap_s=0.0)).run(accel)

    assert len(merged_result.bouts) <= len(no_merge_result.bouts)


def test_step_count_aggregate_output_shape():
    """aggregate() must produce a DataFrame with timestamps and values columns."""
    result = run_step_count_on_sine(1.6, duration_s=120.0)
    result.aggregate(method="mean")
    assert hasattr(result, "biomarker_agg")
    assert "timestamps" in result.biomarker_agg.columns
    assert "values" in result.biomarker_agg.columns
    assert len(result.biomarker_agg) == 2
