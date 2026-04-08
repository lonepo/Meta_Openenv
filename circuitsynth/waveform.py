"""
Waveform analysis for CircuitSynth.

Extracts physics-based metrics from simulated time-domain signals:
  - frequency (FFT + zero-crossing validation)
  - duty cycle
  - amplitude (VOH, VOL, Vpp)
  - rise time and fall time (10%–90%)
  - settling time
  - stability (period variance)
  - waveform similarity to an ideal square wave (normalized MSE)

All metrics are extracted in a simulator-agnostic way from (time, voltage) arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sig_proc


# ---------------------------------------------------------------------------
# WaveformMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class WaveformMetrics:
    """Extracted waveform properties from a transient simulation."""
    frequency:    float     # measured oscillation frequency (Hz)
    duty_cycle:   float     # measured duty cycle [0, 1]
    voh:          float     # output high voltage (V)    — 95th percentile
    vol:          float     # output low  voltage (V)    — 5th  percentile
    vpp:          float     # peak-to-peak voltage (V)
    rise_time:    float     # 10%–90% rise time (s), NaN if unmeasurable
    fall_time:    float     # 90%–10% fall time (s), NaN if unmeasurable
    settling_time: float    # time until waveform stabilises (s)
    stability:    float     # period stability score [0, 1]
    n_cycles:     int       # number of complete cycles in observation window
    sim_success:  bool = True

    # ------------------------------------------------------------------
    @classmethod
    def null(cls) -> "WaveformMetrics":
        """Return a zero/invalid metrics object for non-converged sims."""
        return cls(
            frequency=0.0, duty_cycle=0.0,
            voh=0.0, vol=0.0, vpp=0.0,
            rise_time=float("nan"), fall_time=float("nan"),
            settling_time=float("nan"), stability=0.0,
            n_cycles=0, sim_success=False,
        )

    def to_array(self) -> np.ndarray:
        """Flat float32 array of 10 metrics for RL observation."""
        return np.array([
            self.frequency,
            self.duty_cycle,
            self.voh,
            self.vol,
            self.vpp,
            self.rise_time    if np.isfinite(self.rise_time)     else 0.0,
            self.fall_time    if np.isfinite(self.fall_time)     else 0.0,
            self.settling_time if np.isfinite(self.settling_time) else 0.0,
            self.stability,
            float(self.n_cycles),
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "frequency":     self.frequency,
            "duty_cycle":    self.duty_cycle,
            "voh":           self.voh,
            "vol":           self.vol,
            "vpp":           self.vpp,
            "rise_time":     self.rise_time,
            "fall_time":     self.fall_time,
            "settling_time": self.settling_time,
            "stability":     self.stability,
            "n_cycles":      self.n_cycles,
            "sim_success":   self.sim_success,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_interpolate_crossing(t1: float, v1: float, t2: float, v2: float,
                                  threshold: float) -> float:
    """Find time where linear segment (t1,v1)→(t2,v2) crosses threshold."""
    if abs(v2 - v1) < 1e-20:
        return t1
    frac = (threshold - v1) / (v2 - v1)
    return t1 + frac * (t2 - t1)


def _find_threshold_crossings(t: np.ndarray, v: np.ndarray,
                               threshold: float) -> np.ndarray:
    """
    Return array of times where v crosses threshold (both up and down).
    Uses linear interpolation for sub-sample accuracy.
    """
    above = v >= threshold
    crossing_indices = np.where(np.diff(above.astype(int)) != 0)[0]
    times = []
    for i in crossing_indices:
        tc = _linear_interpolate_crossing(t[i], v[i], t[i + 1], v[i + 1], threshold)
        times.append(tc)
    return np.array(times)


def _find_rising_crossings(t: np.ndarray, v: np.ndarray,
                            threshold: float) -> np.ndarray:
    """Times of upward threshold crossings."""
    above = v >= threshold
    rising = np.where((~above[:-1]) & above[1:])[0]
    times = [_linear_interpolate_crossing(t[i], v[i], t[i+1], v[i+1], threshold)
             for i in rising]
    return np.array(times)


def _find_falling_crossings(t: np.ndarray, v: np.ndarray,
                             threshold: float) -> np.ndarray:
    """Times of downward threshold crossings."""
    above = v >= threshold
    falling = np.where(above[:-1] & (~above[1:]))[0]
    times = [_linear_interpolate_crossing(t[i], v[i], t[i+1], v[i+1], threshold)
             for i in falling]
    return np.array(times)


# ---------------------------------------------------------------------------
# Frequency measurement
# ---------------------------------------------------------------------------

def measure_frequency(t: np.ndarray, v: np.ndarray) -> Tuple[float, int]:
    """
    Measure oscillation frequency using midpoint zero-crossing counting.

    Returns (frequency_hz, n_full_cycles).
    """
    if len(t) < 4:
        return 0.0, 0

    mid = (float(np.percentile(v, 5)) + float(np.percentile(v, 95))) / 2.0
    rising = _find_rising_crossings(t, v, mid)

    if len(rising) < 2:
        # Fall back to FFT
        dt   = float(np.mean(np.diff(t)))
        freqs = np.fft.rfftfreq(len(v), d=dt)
        psd   = np.abs(np.fft.rfft(v - v.mean())) ** 2
        if len(psd) > 1:
            peak_idx  = int(np.argmax(psd[1:])) + 1
            fft_freq  = float(freqs[peak_idx])
            return fft_freq, 0
        return 0.0, 0

    periods = np.diff(rising)
    if len(periods) == 0:
        return 0.0, 0

    mean_period = float(np.mean(periods))
    freq = 1.0 / max(mean_period, 1e-20)
    n_cycles = len(rising) - 1
    return freq, n_cycles


# ---------------------------------------------------------------------------
# Duty cycle
# ---------------------------------------------------------------------------

def measure_duty_cycle(t: np.ndarray, v: np.ndarray) -> float:
    """Measure duty cycle as the fraction of time v is above the midpoint."""
    voh = float(np.percentile(v, 95))
    vol = float(np.percentile(v, 5))
    swing = voh - vol
    if swing < 1e-6:          # flat / constant signal — no meaningful duty cycle
        return 0.0
    mid = (voh + vol) / 2.0
    high_fraction = float(np.mean(v >= mid))
    return float(np.clip(high_fraction, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Amplitude metrics
# ---------------------------------------------------------------------------

def measure_amplitude(v: np.ndarray) -> Tuple[float, float, float]:
    """Return (VOH, VOL, Vpp)."""
    voh = float(np.percentile(v, 95))
    vol = float(np.percentile(v, 5))
    vpp = max(0.0, voh - vol)
    return voh, vol, vpp


# ---------------------------------------------------------------------------
# Rise / fall time (10%–90% of swing)
# ---------------------------------------------------------------------------

def measure_rise_time(t: np.ndarray, v: np.ndarray,
                       voh: float, vol: float) -> float:
    """
    Measure 10%–90% rise time.
    Returns NaN if no valid rising edge is found.
    """
    swing = voh - vol
    if swing < 1e-6:
        return float("nan")

    lo_thresh = vol + 0.10 * swing
    hi_thresh = vol + 0.90 * swing

    rising_lo = _find_rising_crossings(t, v, lo_thresh)
    rising_hi = _find_rising_crossings(t, v, hi_thresh)

    if len(rising_lo) == 0 or len(rising_hi) == 0:
        return float("nan")

    rise_times = []
    for t_lo in rising_lo:
        # Find first hi crossing after t_lo
        candidates = rising_hi[rising_hi > t_lo]
        if len(candidates) > 0:
            rise_times.append(candidates[0] - t_lo)

    return float(np.median(rise_times)) if rise_times else float("nan")


def measure_fall_time(t: np.ndarray, v: np.ndarray,
                       voh: float, vol: float) -> float:
    """
    Measure 90%–10% fall time.
    Returns NaN if no valid falling edge is found.
    """
    swing = voh - vol
    if swing < 1e-6:
        return float("nan")

    hi_thresh = vol + 0.90 * swing
    lo_thresh = vol + 0.10 * swing

    falling_hi = _find_falling_crossings(t, v, hi_thresh)
    falling_lo = _find_falling_crossings(t, v, lo_thresh)

    if len(falling_hi) == 0 or len(falling_lo) == 0:
        return float("nan")

    fall_times = []
    for t_hi in falling_hi:
        candidates = falling_lo[falling_lo > t_hi]
        if len(candidates) > 0:
            fall_times.append(candidates[0] - t_hi)

    return float(np.median(fall_times)) if fall_times else float("nan")


# ---------------------------------------------------------------------------
# Stability (period variance)
# ---------------------------------------------------------------------------

def measure_stability(t: np.ndarray, v: np.ndarray) -> float:
    """
    Compute period stability score in [0, 1].
    1.0 = perfectly regular periods; approaches 0 for chaotic/noisy output.
    """
    mid = (float(np.percentile(v, 5)) + float(np.percentile(v, 95))) / 2.0
    rising = _find_rising_crossings(t, v, mid)

    if len(rising) < 3:
        return 0.0

    periods = np.diff(rising)
    mean_p  = float(np.mean(periods))
    std_p   = float(np.std(periods))

    if mean_p < 1e-20:
        return 0.0

    cv = std_p / mean_p  # coefficient of variation
    return float(np.clip(1.0 - cv, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Settling time
# ---------------------------------------------------------------------------

def measure_settling_time(t: np.ndarray, v: np.ndarray,
                           target_amplitude: float,
                           tolerance: float = 0.10) -> float:
    """
    Estimate the time after which the waveform amplitude remains within
    ±tolerance * target_amplitude.
    Returns NaN if the waveform never settles.
    """
    voh, vol, vpp = measure_amplitude(v)
    if vpp < 1e-6:
        return float("nan")

    # Compute rolling Vpp in 10-cycle windows
    envelope = np.abs(v - np.mean(v))
    band = target_amplitude * tolerance

    # Find first point from which env stays within band
    inside = np.abs(envelope - vpp / 2) < band
    # Require 100 consecutive samples inside band
    window = min(100, len(inside) // 4)
    for i in range(len(inside) - window):
        if inside[i:i + window].all():
            return float(t[i])
    return float("nan")


# ---------------------------------------------------------------------------
# Waveform similarity
# ---------------------------------------------------------------------------

def generate_ideal_square(
    t: np.ndarray,
    frequency: float,
    duty_cycle: float,
    voh: float,
    vol: float,
) -> np.ndarray:
    """Generate an ideal square wave sampled at times t."""
    period = 1.0 / max(frequency, 1e-10)
    phase = (t % period) / period
    return np.where(phase < duty_cycle, voh, vol).astype(np.float64)


def compute_waveform_similarity(
    t: np.ndarray,
    v_sim: np.ndarray,
    target_frequency: float,
    target_duty_cycle: float,
    target_amplitude: float,
) -> float:
    """
    Compute normalized MSE-based waveform similarity against an ideal square wave.

    Returns a score in [0, 1] where 1.0 = perfect match.
    """
    if len(t) < 4 or target_amplitude <= 0:
        return 0.0

    ideal = generate_ideal_square(
        t,
        frequency=target_frequency,
        duty_cycle=target_duty_cycle,
        voh=target_amplitude,
        vol=0.0,
    )
    # Normalize both to [0, 1]
    v_norm = np.clip(v_sim / (target_amplitude + 1e-10), -0.5, 1.5)
    i_norm = ideal / (target_amplitude + 1e-10)

    mse = float(np.mean((v_norm - i_norm) ** 2))
    # mse = 0 → perfect, mse = 1 → completely off
    similarity = float(np.exp(-3.0 * mse))
    return float(np.clip(similarity, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Top-level analysis entry point
# ---------------------------------------------------------------------------

def analyze_waveform(
    time_array: np.ndarray,
    voltage_array: np.ndarray,
    target_frequency: float,
) -> WaveformMetrics:
    """
    Full waveform analysis pipeline.

    Parameters
    ----------
    time_array    : 1-D array of time samples (seconds)
    voltage_array : 1-D array of voltage samples (volts)
    target_frequency : expected frequency (used for settling window selection)

    Returns
    -------
    WaveformMetrics
    """
    t = np.asarray(time_array, dtype=np.float64).ravel()
    v = np.asarray(voltage_array, dtype=np.float64).ravel()

    if len(t) < 10 or len(t) != len(v):
        return WaveformMetrics.null()

    # Discard the first 20% of the trace to skip transient startup
    skip = max(1, len(t) // 5)
    ts = t[skip:]
    vs = v[skip:]

    if len(ts) < 10:
        return WaveformMetrics.null()

    freq, n_cycles    = measure_frequency(ts, vs)
    duty_cycle         = measure_duty_cycle(ts, vs)
    voh, vol, vpp      = measure_amplitude(vs)
    rise_time          = measure_rise_time(ts, vs, voh, vol)
    fall_time          = measure_fall_time(ts, vs, voh, vol)
    stability          = measure_stability(ts, vs)
    settling_time      = measure_settling_time(t, v, vpp)

    return WaveformMetrics(
        frequency=freq,
        duty_cycle=duty_cycle,
        voh=voh,
        vol=vol,
        vpp=vpp,
        rise_time=rise_time,
        fall_time=fall_time,
        settling_time=settling_time,
        stability=stability,
        n_cycles=n_cycles,
        sim_success=True,
    )
