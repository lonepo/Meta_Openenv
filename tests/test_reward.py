"""Unit tests for the decomposed reward function."""

from __future__ import annotations

import numpy as np
import pytest

from circuitsynth.reward import (
    RewardDecomposition,
    RewardWeights,
    TaskTarget,
    compute_reward,
    _amplitude_score,
    _duty_cycle_score,
    _frequency_score,
)
from circuitsynth.waveform import WaveformMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_perfect_metrics(target: TaskTarget) -> WaveformMetrics:
    """Create a WaveformMetrics that perfectly matches the target."""
    return WaveformMetrics(
        frequency=target.frequency,
        duty_cycle=target.duty_cycle,
        voh=target.amplitude,
        vol=0.0,
        vpp=target.amplitude,
        rise_time=1e-5,
        fall_time=1e-5,
        settling_time=1e-3,
        stability=1.0,
        n_cycles=10,
        sim_success=True,
    )


def make_poor_metrics() -> WaveformMetrics:
    """Metrics for a flat, non-oscillating circuit output."""
    return WaveformMetrics(
        frequency=0.0, duty_cycle=0.0,
        voh=0.1, vol=0.0, vpp=0.1,
        rise_time=float("nan"), fall_time=float("nan"),
        settling_time=float("nan"), stability=0.0,
        n_cycles=0, sim_success=True,
    )


DEFAULT_TARGET = TaskTarget(
    frequency=1000.0, duty_cycle=0.50, amplitude=5.0,
    freq_tol=0.10, dc_tol=0.05, amp_tol=0.10,
)


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

class TestSubScores:
    def test_amplitude_score_perfect(self):
        assert _amplitude_score(5.0, 5.0) == pytest.approx(1.0)

    def test_amplitude_score_zero_at_double(self):
        assert _amplitude_score(10.0, 5.0) == pytest.approx(0.0)

    def test_amplitude_score_clamp(self):
        assert 0.0 <= _amplitude_score(-1.0, 5.0) <= 1.0

    def test_frequency_score_perfect(self):
        s = _frequency_score(1000.0, 1000.0)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_frequency_score_monotone(self):
        base = _frequency_score(1000.0, 1000.0)
        s10  = _frequency_score(1100.0, 1000.0)
        s50  = _frequency_score(1500.0, 1000.0)
        assert base >= s10 >= s50

    def test_duty_cycle_score_perfect(self):
        assert _duty_cycle_score(0.50, 0.50) == pytest.approx(1.0)

    def test_duty_cycle_score_zero_at_0_5_error(self):
        assert _duty_cycle_score(1.0, 0.50) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Full reward function
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_perfect_circuit_gets_high_reward(self):
        m = make_perfect_metrics(DEFAULT_TARGET)
        d = compute_reward(
            metrics=m, target=DEFAULT_TARGET,
            sim_success=True, n_components=8, max_components=12,
            n_invalid_actions=0, max_steps=30, n_convergence_failures=0,
            weights=RewardWeights(), over_budget=False,
        )
        assert d.total >= 0.5, f"Perfect circuit should get reward ≥ 0.5, got {d.total:.4f}"
        assert d.all_thresholds_met is True

    def test_perfect_circuit_bonus_triggers(self):
        m = make_perfect_metrics(DEFAULT_TARGET)
        d = compute_reward(
            metrics=m, target=DEFAULT_TARGET,
            sim_success=True, n_components=8, max_components=12,
            n_invalid_actions=0, max_steps=30, n_convergence_failures=0,
            weights=RewardWeights(), over_budget=False,
        )
        assert d.terminal_bonus > 0

    def test_sim_failure_gives_low_reward(self):
        m = WaveformMetrics.null()
        d = compute_reward(
            metrics=m, target=DEFAULT_TARGET,
            sim_success=False, n_components=8, max_components=12,
            n_invalid_actions=0, max_steps=30, n_convergence_failures=3,
            weights=RewardWeights(), over_budget=False,
        )
        assert d.total <= 0.0

    def test_poor_metrics_gets_low_reward(self):
        m = make_poor_metrics()
        d = compute_reward(
            metrics=m, target=DEFAULT_TARGET,
            sim_success=True, n_components=10, max_components=12,
            n_invalid_actions=5, max_steps=30, n_convergence_failures=0,
            weights=RewardWeights(), over_budget=False,
        )
        assert d.total < 0.20

    def test_reward_in_valid_range(self):
        for _ in range(20):
            rng = np.random.default_rng(42)
            m = WaveformMetrics(
                frequency=float(rng.uniform(0, 5000)),
                duty_cycle=float(rng.uniform(0, 1)),
                voh=float(rng.uniform(0, 10)),
                vol=0.0,
                vpp=float(rng.uniform(0, 10)),
                rise_time=float(rng.uniform(1e-6, 1e-3)),
                fall_time=float(rng.uniform(1e-6, 1e-3)),
                settling_time=float(rng.uniform(1e-4, 1e-2)),
                stability=float(rng.uniform(0, 1)),
                n_cycles=int(rng.integers(0, 20)),
                sim_success=True,
            )
            d = compute_reward(
                metrics=m, target=DEFAULT_TARGET,
                sim_success=True, n_components=8, max_components=12,
                n_invalid_actions=0, max_steps=30, n_convergence_failures=0,
                weights=RewardWeights(), over_budget=False,
            )
            assert -1.0 <= d.total <= 1.0, (
                f"Reward out of range: {d.total} for metrics {m}"
            )

    def test_over_budget_adds_penalty(self):
        m = make_perfect_metrics(DEFAULT_TARGET)
        d_ok  = compute_reward(m, DEFAULT_TARGET, True, 8, 10, 0, 30, 0, RewardWeights(), False)
        d_over= compute_reward(m, DEFAULT_TARGET, True, 8, 10, 0, 30, 0, RewardWeights(), True)
        assert d_ok.total >= d_over.total

    def test_invalid_actions_reduce_reward(self):
        m = make_perfect_metrics(DEFAULT_TARGET)
        d_clean = compute_reward(m, DEFAULT_TARGET, True, 8, 12, 0, 30, 0, RewardWeights(), False)
        d_dirty = compute_reward(m, DEFAULT_TARGET, True, 8, 12, 10, 30, 0, RewardWeights(), False)
        assert d_clean.total >= d_dirty.total

    def test_decomposition_to_dict_contains_all_keys(self):
        m = make_perfect_metrics(DEFAULT_TARGET)
        d = compute_reward(m, DEFAULT_TARGET, True, 8, 12, 0, 30, 0, RewardWeights(), False)
        dct = d.to_dict()
        required = ["total", "waveform_similarity", "amplitude_score", "frequency_score",
                    "duty_cycle_score", "stability_score", "simulation_success_score",
                    "component_penalty", "invalid_action_penalty",
                    "convergence_penalty", "terminal_bonus", "all_thresholds_met"]
        for k in required:
            assert k in dct, f"Missing key in decomposition: {k}"

    def test_reward_weights_easy_vs_hard(self):
        """Hard task weights should be stricter (higher penalties)."""
        w_easy = RewardWeights.easy()
        w_hard = RewardWeights.hard()
        assert w_hard.component_penalty >= w_easy.component_penalty
        assert w_hard.terminal_bonus >= w_easy.terminal_bonus
