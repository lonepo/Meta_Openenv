"""Unit tests for waveform analysis functions."""

from __future__ import annotations

import numpy as np
import pytest

from circuitsynth.waveform import (
    WaveformMetrics,
    analyze_waveform,
    compute_waveform_similarity,
    generate_ideal_square,
    measure_amplitude,
    measure_duty_cycle,
    measure_frequency,
    measure_rise_time,
    measure_fall_time,
    measure_stability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_square_wave(
    freq: float = 1000.0,
    duty: float = 0.50,
    voh: float = 5.0,
    vol: float = 0.0,
    n_cycles: int = 10,
    n_pts_per_cycle: int = 200,
    noise_std: float = 0.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean (or slightly noisy) square wave."""
    period = 1.0 / freq
    n_pts  = n_cycles * n_pts_per_cycle
    t = np.linspace(0, n_cycles * period, n_pts, endpoint=False)
    phase = (t % period) / period
    v = np.where(phase < duty, voh, vol).astype(np.float64)
    if noise_std > 0:
        rng = np.random.default_rng(rng_seed)
        v += rng.normal(0, noise_std, len(v))
    return t, v


# ---------------------------------------------------------------------------
# Frequency measurement
# ---------------------------------------------------------------------------

class TestFrequencyMeasurement:
    @pytest.mark.parametrize("freq", [100.0, 555.0, 1000.0, 2000.0, 5000.0])
    def test_clean_square_wave_frequency(self, freq):
        t, v = make_square_wave(freq=freq, n_cycles=20)
        measured, n_cycles = measure_frequency(t, v)
        assert abs(measured - freq) / freq < 0.02, (
            f"Frequency error > 2% for {freq} Hz: measured {measured:.1f} Hz"
        )
        assert n_cycles >= 5

    def test_flat_signal_returns_zero(self):
        t = np.linspace(0, 0.01, 1000)
        v = np.full_like(t, 2.5)
        freq, n = measure_frequency(t, v)
        # Accept near-zero or small FFT artefact
        assert freq < 50.0 or n == 0

    def test_noisy_square_wave(self):
        t, v = make_square_wave(freq=1000.0, n_cycles=20, noise_std=0.2)
        measured, _ = measure_frequency(t, v)
        assert abs(measured - 1000.0) / 1000.0 < 0.10


# ---------------------------------------------------------------------------
# Duty cycle
# ---------------------------------------------------------------------------

class TestDutyCycle:
    @pytest.mark.parametrize("duty", [0.25, 0.50, 0.75])
    def test_duty_cycle_accuracy(self, duty):
        t, v = make_square_wave(duty=duty, n_cycles=10)
        measured = measure_duty_cycle(t, v)
        assert abs(measured - duty) < 0.03, f"DC error: {abs(measured - duty):.3f}"

    def test_flat_high_signal(self):
        # A constant-voltage signal has no oscillation swing → DC = 0.0 (undefined/noisy)
        t = np.linspace(0, 0.01, 1000)
        v = np.full_like(t, 5.0)
        dc = measure_duty_cycle(t, v)
        assert dc == pytest.approx(0.0)    # flat signal → swing < threshold → returns 0

    def test_flat_low_signal(self):
        t = np.linspace(0, 0.01, 1000)
        v = np.zeros_like(t)
        dc = measure_duty_cycle(t, v)
        assert dc == pytest.approx(0.0)    # flat signal → no oscillation


# ---------------------------------------------------------------------------
# Amplitude
# ---------------------------------------------------------------------------

class TestAmplitude:
    def test_clean_square_wave_amplitude(self):
        t, v = make_square_wave(voh=5.0, vol=0.0, n_cycles=10)
        voh, vol, vpp = measure_amplitude(v)
        assert voh == pytest.approx(5.0, abs=0.1)
        assert vol == pytest.approx(0.0, abs=0.1)
        assert vpp == pytest.approx(5.0, abs=0.2)

    def test_shifted_square_wave(self):
        t, v = make_square_wave(voh=3.3, vol=1.0, n_cycles=10)
        voh, vol, vpp = measure_amplitude(v)
        assert vpp == pytest.approx(2.3, abs=0.2)


# ---------------------------------------------------------------------------
# Rise / fall time
# ---------------------------------------------------------------------------

class TestEdgeTimes:
    def test_rise_time_is_finite_for_clean_wave(self):
        t, v = make_square_wave(freq=1000.0, n_cycles=10)
        voh, vol, _ = measure_amplitude(v)
        rt = measure_rise_time(t, v, voh, vol)
        assert np.isfinite(rt), "Rise time should be finite"
        # Rise time should be < 5% of one period
        assert rt < 0.05 / 1000.0

    def test_fall_time_is_finite(self):
        t, v = make_square_wave(freq=1000.0, n_cycles=10)
        voh, vol, _ = measure_amplitude(v)
        ft = measure_fall_time(t, v, voh, vol)
        assert np.isfinite(ft)

    def test_flat_signal_returns_nan(self):
        t = np.linspace(0, 0.01, 1000)
        v = np.full_like(t, 2.5)
        rt = measure_rise_time(t, v, 2.5, 2.5)
        assert not np.isfinite(rt)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

class TestStability:
    def test_perfect_wave_has_high_stability(self):
        t, v = make_square_wave(n_cycles=20)
        stab = measure_stability(t, v)
        assert stab >= 0.95

    def test_noisy_wave_has_lower_stability(self):
        t_clean, v_clean = make_square_wave(n_cycles=20)
        t_noisy, v_noisy = make_square_wave(n_cycles=20, noise_std=0.5)
        s_clean = measure_stability(t_clean, v_clean)
        s_noisy = measure_stability(t_noisy, v_noisy)
        assert s_clean >= s_noisy

    def test_flat_signal_returns_zero_stability(self):
        t = np.linspace(0, 0.01, 1000)
        v = np.zeros_like(t)
        assert measure_stability(t, v) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Waveform similarity
# ---------------------------------------------------------------------------

class TestWaveformSimilarity:
    def test_perfect_match_scores_near_1(self):
        t, v = make_square_wave(freq=1000.0, duty=0.50, voh=5.0, vol=0.0, n_cycles=10)
        score = compute_waveform_similarity(t, v, 1000.0, 0.50, 5.0)
        assert score >= 0.85, f"Perfect match should score ≥ 0.85, got {score:.3f}"

    def test_flat_signal_scores_near_0(self):
        t = np.linspace(0, 0.01, 1000)
        v = np.zeros_like(t)
        score = compute_waveform_similarity(t, v, 1000.0, 0.50, 5.0)
        assert score < 0.5

    def test_wrong_frequency_has_lower_score_than_correct(self):
        t, v = make_square_wave(freq=1000.0, n_cycles=10)
        s_correct = compute_waveform_similarity(t, v, 1000.0, 0.50, 5.0)
        s_wrong   = compute_waveform_similarity(t, v, 500.0,  0.50, 5.0)
        assert s_correct >= s_wrong


# ---------------------------------------------------------------------------
# Full analyze_waveform pipeline
# ---------------------------------------------------------------------------

class TestAnalyzeWaveform:
    def test_returns_waveform_metrics_type(self):
        t, v = make_square_wave(freq=1000.0, n_cycles=15)
        m = analyze_waveform(t, v, target_frequency=1000.0)
        assert isinstance(m, WaveformMetrics)
        assert m.sim_success

    def test_null_for_empty_array(self):
        m = analyze_waveform(np.array([]), np.array([]), 1000.0)
        assert not m.sim_success

    def test_frequency_extracted_correctly(self):
        t, v = make_square_wave(freq=555.0, n_cycles=20)
        m = analyze_waveform(t, v, target_frequency=555.0)
        assert abs(m.frequency - 555.0) / 555.0 < 0.05

    def test_to_array_has_correct_shape(self):
        t, v = make_square_wave(n_cycles=10)
        m = analyze_waveform(t, v, 1000.0)
        arr = m.to_array()
        assert arr.shape == (10,)
        assert arr.dtype == np.float32
