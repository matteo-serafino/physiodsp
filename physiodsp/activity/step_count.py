from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from scipy.signal import butter, sosfiltfilt, welch

from physiodsp.base import BaseAlgorithm
from physiodsp.sensors.imu.accelerometer import AccelerometerData


@dataclass
class StepBout:
    """A continuous bout of detected locomotion."""
    start_s: float
    end_s: float
    step_count: int
    mean_cadence_spm: float
    activity: str  # "walk" | "run"
    step_timestamps: List[float] = field(default_factory=list)


class StepCountSettings(BaseModel):
    """Configuration for the StepCount algorithm.

    All thresholds are expressed in SI-consistent units (g for acceleration,
    Hz for frequency, seconds for time).
    """

    # --- Stage 1: Pre-processing ---
    hp_cutoff_hz: PositiveFloat = Field(
        default=0.5,
        description="High-pass Butterworth cutoff in Hz for gravity removal.",
    )

    # --- Stage 2: Motion gate ---
    sma_sleep_threshold: PositiveFloat = Field(
        default=0.02,
        description="SMA below this value (g) -> window classified as sleep/rest.",
    )
    sma_adl_threshold: PositiveFloat = Field(
        default=0.05,
        description="SMA above this value (g) is treated as clear locomotion energy.",
    )

    # --- Stage 3: Periodicity detector ---
    periodicity_threshold: PositiveFloat = Field(
        default=0.50,
        description="Base autocorrelation peak threshold for locomotion classification.",
    )
    spectral_purity_threshold: PositiveFloat = Field(
        default=8.0,
        description="PSD peak-to-mean ratio used for secondary confirmation.",
    )

    # --- Stage 4: Peak detector ---
    threshold_factor: PositiveFloat = Field(
        default=0.30,
        description="Step peak threshold = threshold_factor * local RMS amplitude.",
    )
    analysis_window_s: PositiveFloat = Field(
        default=2.0,
        description="Window length in seconds for processing.",
    )

    # --- Stage 5: Post-processing ---
    min_bout_steps: PositiveInt = Field(
        default=3,
        description="Minimum steps required for a bout to be retained.",
    )
    bout_merge_gap_s: float = Field(
        default=3.0,
        ge=0.0,
        description="Merge consecutive bouts whose inter-bout gap is shorter than this.",
    )
    interpolation_max_void_s: PositiveFloat = Field(
        default=2.0,
        description="Maximum gap duration (seconds) for missing step interpolation.",
    )

    # --- Output / aggregation ---
    aggregation_window: PositiveInt = Field(
        default=60,
        description="Aggregation window length in seconds for aggregate().",
    )


class StepCount(BaseAlgorithm):
    """Wrist-accelerometer step counting algorithm.

    Five-stage pipeline:
      1. Pre-processing  - VM, high-pass filter
      2. Motion gate     - SMA threshold (sleep / rest rejection)
      3. Periodicity     - autocorrelation + PSD to confirm locomotion
      4. Peak detection  - adaptive-threshold peak detector on BP signal
      5. Validation      - bout merging, short-bout filter, step interpolation
    """

    _algorithm_name = "StepCount"
    _version = "0.1.0"

    def __init__(self, settings: StepCountSettings = StepCountSettings()) -> None:
        self.settings = settings

    def run(self, accelerometer: AccelerometerData) -> "StepCount":
        """Run the full step counting pipeline on tri-axial accelerometer data."""
        cfg = self.settings
        fs = float(accelerometer.fs)

        # ----------------------------------------------------------------
        # Stage 1 — Pre-processing
        # ----------------------------------------------------------------
        # Vector magnitude (includes gravity)
        vm = np.sqrt(accelerometer.x**2 + accelerometer.y**2 + accelerometer.z**2)

        # Remove gravity — high-pass at hp_cutoff_hz
        vm_dynamic = highpass_filter(vm, fs=fs, cutoff_hz=cfg.hp_cutoff_hz)

        t = np.asarray(accelerometer.timestamps, dtype=float)
        total_duration_s = t[-1] - t[0] if len(t) > 1 else 0.0

        # ----------------------------------------------------------------
        # Stage 2–4 — Window-by-window processing
        # ----------------------------------------------------------------
        win_samples = int(cfg.analysis_window_s * fs)
        n_windows = len(vm_dynamic) // win_samples

        all_step_times: List[float] = []
        all_step_cadences: List[float] = []
        all_step_activities: List[str] = []
        window_activity: List[str] = []
        window_timestamps: List[float] = []

        # --- Stage 2–3: classify every window (motion gate + periodicity) ---
        # Detection is deferred to Stage 4 so that contiguous locomotion windows
        # can be filtered and peak-picked as one continuous signal. Detecting
        # each 2 s window in isolation drops steps at window boundaries
        # (filter edge transients + per-window detector reset), which
        # systematically underestimates the count during sustained walking.
        window_is_loco: List[bool] = []
        window_cadence_hz: List[float] = []
        prev_cadence_hz: float = 0.0

        for w in range(n_windows):
            s = w * win_samples
            e = s + win_samples
            seg = vm_dynamic[s:e]
            window_timestamps.append(t[s])

            # --- Stage 2: SMA gate ---
            sma = float(np.mean(np.abs(seg)))

            # --- Stage 3: Periodicity / locomotion classifier ---
            is_loco, cadence_hz, activity_state = is_locomotion_window(
                dynamic_vm=seg,
                fs=fs,
                sma=sma,
                sma_sleep_thr=cfg.sma_sleep_threshold,
                sma_adl_thr=cfg.sma_adl_threshold,
                periodicity_threshold=cfg.periodicity_threshold,
                spectral_purity_threshold=cfg.spectral_purity_threshold,
            )

            if is_loco:
                if cadence_hz <= 0.0:
                    cadence_hz = prev_cadence_hz * 0.9
                if cadence_hz <= 0.0:
                    # No usable cadence estimate: demote to rest.
                    is_loco = False
                    activity_state = "rest"
                else:
                    if prev_cadence_hz > 0.0:
                        cadence_hz = 0.7 * prev_cadence_hz + 0.3 * cadence_hz
                    prev_cadence_hz = cadence_hz

            if not is_loco:
                prev_cadence_hz = 0.0

            window_activity.append(activity_state)
            window_is_loco.append(is_loco)
            window_cadence_hz.append(cadence_hz if is_loco else 0.0)

        # --- Stage 4: continuous detection over contiguous locomotion runs ---
        for run_start, run_end, run_cadence_hz in group_locomotion_runs(
            window_is_loco=window_is_loco,
            window_cadence_hz=window_cadence_hz,
            win_samples=win_samples,
            n_samples=len(vm_dynamic),
        ):
            seg = vm_dynamic[run_start:run_end]
            bp_seg = adaptive_bandpass_filter(
                seg, center_freq_hz=run_cadence_hz, fs=fs
            )
            peak_indices = detect_steps_in_window(
                bp_signal=bp_seg,
                fs=fs,
                cadence_hz=run_cadence_hz,
                threshold_factor=cfg.threshold_factor,
            )

            cadence_spm = run_cadence_hz * 60.0
            for idx in peak_indices:
                abs_idx = run_start + idx
                win_idx = min(abs_idx // win_samples, n_windows - 1)
                all_step_times.append(float(t[abs_idx]))
                all_step_cadences.append(cadence_spm)
                all_step_activities.append(window_activity[win_idx])

        # ----------------------------------------------------------------
        # Stage 5 — Post-processing
        # ----------------------------------------------------------------
        if len(all_step_times) == 0:
            self._set_empty_outputs(t, total_duration_s, window_timestamps,
                                    window_activity)
            return self

        step_times = np.array(all_step_times)
        step_cadences = np.array(all_step_cadences)
        step_activities = np.array(all_step_activities)

        mean_cadence_hz = float(np.mean(step_cadences)) / 60.0
        step_times_interp = interpolate_missed_steps(
            step_times_s=step_times.tolist(),
            cadence_hz=mean_cadence_hz,
            max_void_s=cfg.interpolation_max_void_s,
        )
        step_times = np.array(step_times_interp)
        step_cadences = np.full(len(step_times), np.mean(step_cadences))
        step_activities = np.full(len(step_times), "walk", dtype=object)

        bouts = group_steps_into_bouts(
            step_times_s=step_times,
            cadence_per_step=step_cadences,
            activities=step_activities,
            max_gap_s=cfg.bout_merge_gap_s if cfg.bout_merge_gap_s > 0 else 1e9,
        )

        bouts = merge_close_bouts(bouts, merge_gap_s=cfg.bout_merge_gap_s)
        bouts = filter_short_bouts(bouts, min_steps=cfg.min_bout_steps)

        surviving_steps = sorted(
            t for b in bouts for t in b.step_timestamps
        )

        self.bouts: List[StepBout] = bouts
        self.total_steps: int = len(surviving_steps)
        self.step_timestamps: np.ndarray = np.array(surviving_steps)

        t0 = float(t[0])
        relative_steps = self.step_timestamps - t0
        bin_ts, cadence_spm = steps_to_cadence_series(
            step_times_s=relative_steps,
            total_duration_s=total_duration_s,
            bin_s=1.0,
        )

        activity_per_bin = self._map_activity_to_bins(
            window_timestamps=window_timestamps,
            window_activity=window_activity,
            bin_timestamps=bin_ts + t0,
        )

        step_count_per_bin = np.histogram(
            relative_steps,
            bins=np.append(bin_ts, bin_ts[-1] + 1.0),
        )[0]

        self.biomarker = pd.DataFrame({
            "timestamps": bin_ts + t0,
            "step_count": step_count_per_bin,
            "cadence_spm": cadence_spm,
            "activity_state": activity_per_bin,
        })

        self.timestamps = self.biomarker["timestamps"].values
        self.values = self.biomarker["cadence_spm"].values

        return self

    def aggregate(self, method: str = "mean") -> "StepCount":
        """Aggregate cadence_spm over aggregation_window-second bins."""
        super().aggregate(self.timestamps, self.values, method)
        return self

    def _set_empty_outputs(
        self,
        t: np.ndarray,
        total_duration_s: float,
        window_timestamps: List[float],
        window_activity: List[str],
    ) -> None:
        self.bouts = []
        self.total_steps = 0
        self.step_timestamps = np.array([])
        t0 = float(t[0]) if len(t) > 0 else 0.0
        n_bins = max(1, int(total_duration_s))
        bin_ts = np.arange(n_bins, dtype=float) + t0
        activity_per_bin = self._map_activity_to_bins(
            window_timestamps, window_activity, bin_ts
        )
        self.biomarker = pd.DataFrame({
            "timestamps": bin_ts,
            "step_count": np.zeros(n_bins, dtype=int),
            "cadence_spm": np.zeros(n_bins, dtype=float),
            "activity_state": activity_per_bin,
        })
        self.timestamps = self.biomarker["timestamps"].values
        self.values = self.biomarker["cadence_spm"].values

    @staticmethod
    def _map_activity_to_bins(
        window_timestamps: List[float],
        window_activity: List[str],
        bin_timestamps: np.ndarray,
    ) -> np.ndarray:
        result = np.full(len(bin_timestamps), "rest", dtype=object)
        if not window_timestamps:
            return result
        wt = np.array(window_timestamps)
        for i, bt in enumerate(bin_timestamps):
            idx = int(np.argmin(np.abs(wt - bt)))
            result[i] = window_activity[idx]
        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def highpass_filter(signal: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, signal)


def adaptive_bandpass_filter(
    signal: np.ndarray,
    center_freq_hz: float,
    fs: float,
    order: int = 4,
    width_factor: float = 0.5,
) -> np.ndarray:
    nyq = fs / 2.0
    low = max(0.5, center_freq_hz * (1.0 - width_factor))
    high = min(nyq - 0.1, center_freq_hz * (1.0 + width_factor))
    if low >= high:
        low, high = 0.5, min(nyq - 0.1, 3.5)
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def autocorrelation_peak(
    signal: np.ndarray,
    fs: float,
    freq_min_hz: float = 0.5,
    freq_max_hz: float = 3.5,
) -> Tuple[float, float]:
    sig = signal - signal.mean()
    if np.std(sig) < 1e-9:
        return 0.0, 0.0

    acf = np.correlate(sig, sig, mode="full")
    acf = acf[len(acf) // 2:]
    acf = acf / (acf[0] + 1e-12)

    lag_min = max(1, int(fs / freq_max_hz))
    lag_max = min(len(acf) - 1, int(fs / freq_min_hz))

    if lag_min >= lag_max:
        return 0.0, 0.0

    sub = acf[lag_min:lag_max + 1]
    best_rel = int(np.argmax(sub))
    best_lag = lag_min + best_rel
    peak_val = float(acf[best_lag])
    dominant_freq = fs / best_lag if best_lag > 0 else 0.0
    return peak_val, dominant_freq


def psd_dominant_freq(
    signal: np.ndarray,
    fs: float,
    freq_min_hz: float = 0.5,
    freq_max_hz: float = 4.0,
) -> Tuple[float, float]:
    nperseg = min(len(signal), 256)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
    if not np.any(mask):
        return 0.0, 0.0

    sub_freqs = freqs[mask]
    sub_psd = psd[mask]
    peak_idx = int(np.argmax(sub_psd))
    dominant_freq = float(sub_freqs[peak_idx])
    spectral_purity = float(sub_psd[peak_idx] / (np.mean(sub_psd) + 1e-12))
    return dominant_freq, spectral_purity


def is_locomotion_window(
    dynamic_vm: np.ndarray,
    fs: float,
    sma: float,
    sma_sleep_thr: float,
    sma_adl_thr: float,
    periodicity_threshold: float,
    spectral_purity_threshold: float,
) -> Tuple[bool, float, str]:
    if sma < sma_sleep_thr:
        return False, 0.0, "sleep"

    if sma > sma_adl_thr:
        eff_threshold = periodicity_threshold * 0.85
    else:
        eff_threshold = periodicity_threshold * 1.15

    ac_peak, dominant_freq = autocorrelation_peak(dynamic_vm, fs=fs)

    if ac_peak >= eff_threshold and dominant_freq > 0.0:
        activity = "run" if dominant_freq > 2.2 else "walk"
        return True, dominant_freq, activity

    ambiguous_low = eff_threshold * 0.70
    if ambiguous_low <= ac_peak < eff_threshold:
        psd_freq, purity = psd_dominant_freq(dynamic_vm, fs=fs)
        if purity >= spectral_purity_threshold and psd_freq > 0.0:
            activity = "run" if psd_freq > 2.2 else "walk"
            return True, psd_freq, activity

    return False, 0.0, "rest" if sma > sma_sleep_thr else "sleep"


def group_locomotion_runs(
    window_is_loco: List[bool],
    window_cadence_hz: List[float],
    win_samples: int,
    n_samples: int,
) -> List[Tuple[int, int, float]]:
    """Group consecutive locomotion windows into continuous sample-index runs.

    Returns a list of ``(start_sample, end_sample, mean_cadence_hz)`` tuples,
    one per contiguous block of locomotion windows. Detecting steps over these
    runs (rather than per fixed window) avoids losing steps to filter edge
    effects and detector resets at every 2 s boundary.

    The final run is extended to ``n_samples`` so the trailing remainder
    samples that the integer window count drops are still searched for steps.
    """
    runs: List[Tuple[int, int, float]] = []
    n_windows = len(window_is_loco)
    i = 0
    while i < n_windows:
        if not window_is_loco[i]:
            i += 1
            continue
        j = i
        cadences: List[float] = []
        while j < n_windows and window_is_loco[j]:
            cadences.append(window_cadence_hz[j])
            j += 1
        start = i * win_samples
        end = n_samples if j == n_windows else j * win_samples
        runs.append((start, end, float(np.mean(cadences))))
        i = j
    return runs


def detect_steps_in_window(
    bp_signal: np.ndarray,
    fs: float,
    cadence_hz: float,
    threshold_factor: float = 0.30,
    cadence_tolerance: float = 0.60,
) -> List[int]:
    if len(bp_signal) == 0 or cadence_hz <= 0:
        return []

    rms_win = max(1, int(fs * 2.0))
    sq = bp_signal ** 2
    kernel = np.ones(rms_win) / rms_win
    rms_smooth = np.sqrt(np.convolve(sq, kernel, mode="same") + 1e-12)
    threshold = threshold_factor * rms_smooth

    min_gap = max(1, int((cadence_tolerance / cadence_hz) * fs))
    max_gap = int((2.0 / (cadence_hz * cadence_tolerance)) * fs)

    steps: List[int] = []
    last_step_idx = -max_gap
    in_peak = False
    peak_val = -np.inf
    peak_idx = 0

    for i, (val, thr) in enumerate(zip(bp_signal, threshold)):
        if val > thr:
            if not in_peak:
                in_peak = True
                peak_val = val
                peak_idx = i
            elif val > peak_val:
                peak_val = val
                peak_idx = i
        else:
            if in_peak:
                in_peak = False
                gap = peak_idx - last_step_idx
                if not steps:
                    # Always accept the first peak to start the sequence
                    steps.append(peak_idx)
                    last_step_idx = peak_idx
                elif min_gap <= gap <= max_gap:
                    steps.append(peak_idx)
                    last_step_idx = peak_idx
                elif gap > max_gap:
                    # Possible missed step(s) - still accept this peak
                    steps.append(peak_idx)
                    last_step_idx = peak_idx

    if in_peak:
        gap = peak_idx - last_step_idx
        if min_gap <= gap:
            steps.append(peak_idx)

    return steps


def group_steps_into_bouts(
    step_times_s: np.ndarray,
    cadence_per_step: np.ndarray,
    activities: np.ndarray,
    max_gap_s: float = 3.0,
) -> List[StepBout]:
    if len(step_times_s) == 0:
        return []

    bouts: List[StepBout] = []
    bout_steps = [step_times_s[0]]
    bout_cadences = [cadence_per_step[0]]
    bout_activities = [activities[0]]

    for i in range(1, len(step_times_s)):
        gap = step_times_s[i] - step_times_s[i - 1]
        if gap <= max_gap_s:
            bout_steps.append(step_times_s[i])
            bout_cadences.append(cadence_per_step[i])
            bout_activities.append(activities[i])
        else:
            bouts.append(_make_bout(bout_steps, bout_cadences, bout_activities))
            bout_steps = [step_times_s[i]]
            bout_cadences = [cadence_per_step[i]]
            bout_activities = [activities[i]]

    bouts.append(_make_bout(bout_steps, bout_cadences, bout_activities))
    return bouts


def _make_bout(
    step_times: List[float],
    cadences: List[float],
    activities: List[str],
) -> StepBout:
    dominant_activity = max(set(activities), key=activities.count)
    return StepBout(
        start_s=step_times[0],
        end_s=step_times[-1],
        step_count=len(step_times),
        mean_cadence_spm=float(np.mean(cadences)),
        activity=dominant_activity,
        step_timestamps=list(step_times),
    )


def filter_short_bouts(bouts: List[StepBout], min_steps: int = 3) -> List[StepBout]:
    return [b for b in bouts if b.step_count >= min_steps]


def merge_close_bouts(bouts: List[StepBout], merge_gap_s: float = 3.0) -> List[StepBout]:
    if not bouts or merge_gap_s <= 0:
        return bouts

    merged: List[StepBout] = [bouts[0]]
    for current in bouts[1:]:
        prev = merged[-1]
        gap = current.start_s - prev.end_s
        if gap <= merge_gap_s:
            all_steps = prev.step_timestamps + current.step_timestamps
            all_cadences = [prev.mean_cadence_spm] * prev.step_count + [
                current.mean_cadence_spm] * current.step_count
            all_acts = [prev.activity] * prev.step_count + \
                       [current.activity] * current.step_count
            merged[-1] = _make_bout(all_steps, all_cadences, all_acts)
        else:
            merged.append(current)
    return merged


def interpolate_missed_steps(
    step_times_s: List[float],
    cadence_hz: float,
    max_void_s: float = 2.0,
) -> List[float]:
    if len(step_times_s) < 2 or cadence_hz <= 0:
        return list(step_times_s)

    expected_period_s = 1.0 / cadence_hz
    result = [step_times_s[0]]

    for i in range(1, len(step_times_s)):
        gap = step_times_s[i] - step_times_s[i - 1]
        if gap > expected_period_s * 1.3 and gap <= max_void_s:
            n_missing = round(gap / expected_period_s) - 1
            if n_missing >= 1:
                for k in range(1, n_missing + 1):
                    interp_t = step_times_s[i - 1] + k * expected_period_s
                    result.append(interp_t)
        result.append(step_times_s[i])

    return sorted(result)


def steps_to_cadence_series(
    step_times_s: np.ndarray,
    total_duration_s: float,
    bin_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = max(1, int(np.ceil(total_duration_s / bin_s)))
    bin_edges = np.arange(0, n_bins + 1) * bin_s
    counts, _ = np.histogram(step_times_s, bins=bin_edges)
    cadence_spm = counts * (60.0 / bin_s)
    bin_timestamps = bin_edges[:-1]
    return bin_timestamps, cadence_spm
