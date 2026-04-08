"""
Decomposed reward function for CircuitSynth.

Implements a weighted, interpretable reward in [-1, 1] composed of:
  + waveform_similarity   (normalized MSE vs ideal square wave)
  + amplitude_score       (Vpp closeness to target)
  + frequency_score       (exponential decay with relative error)
  + duty_cycle_score      (linear penalty on |dc - target_dc|)
  + stability_score       (period regularity)
  + simulation_success    (binary: did sim converge?)
  − component_penalty     (encourages minimal design)
  − invalid_action_penalty(discourages illegal wiring)
  − convergence_penalty   (penalises SPICE failures)
  − overbudget_penalty    (over max_components)
  + terminal_bonus        (when ALL thresholds simultaneously met)

Each sub-score is individually clamped to [0, 1] before weighting.
The weighted total is clamped to [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .waveform import WaveformMetrics, compute_waveform_similarity
from .utils import exponential_score


# ---------------------------------------------------------------------------
# Reward weight config
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """Per-component weights for the reward function. Should sum to ≈1."""
    waveform_similarity:    float = 0.25
    amplitude:              float = 0.20
    frequency:              float = 0.20
    duty_cycle:             float = 0.15
    stability:              float = 0.10
    simulation_success:     float = 0.05
    component_penalty:      float = 0.05
    invalid_action:         float = 0.03
    convergence:            float = 0.02
    overbudget:             float = 0.02
    terminal_bonus:         float = 0.20   # bonus if ALL thresholds met simultaneously

    # ------------------------------------------------------------------
    @classmethod
    def easy(cls) -> "RewardWeights":
        return cls(
            waveform_similarity=0.20, amplitude=0.15, frequency=0.20,
            duty_cycle=0.10, stability=0.08, simulation_success=0.07,
            component_penalty=0.03, invalid_action=0.02,
            convergence=0.02, overbudget=0.01, terminal_bonus=0.15,
        )

    @classmethod
    def medium(cls) -> "RewardWeights":
        return cls()  # default

    @classmethod
    def hard(cls) -> "RewardWeights":
        return cls(
            waveform_similarity=0.30, amplitude=0.22, frequency=0.22,
            duty_cycle=0.16, stability=0.12, simulation_success=0.04,
            component_penalty=0.08, invalid_action=0.04,
            convergence=0.03, overbudget=0.05, terminal_bonus=0.25,
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Reward decomposition result
# ---------------------------------------------------------------------------

@dataclass
class RewardDecomposition:
    """Full breakdown of reward components for logging and debugging."""
    total:                   float

    # Positive sub-scores (weighted)
    waveform_similarity:     float = 0.0
    amplitude_score:         float = 0.0
    frequency_score:         float = 0.0
    duty_cycle_score:        float = 0.0
    stability_score:         float = 0.0
    simulation_success_score: float = 0.0

    # Penalties (negative, weighted)
    component_penalty:       float = 0.0
    invalid_action_penalty:  float = 0.0
    convergence_penalty:     float = 0.0
    overbudget_penalty:      float = 0.0

    # Terminal bonus
    terminal_bonus:          float = 0.0
    all_thresholds_met:      bool  = False

    # Raw unweighted sub-scores for diagnostics
    raw_waveform_sim:        float = 0.0
    raw_amplitude:           float = 0.0
    raw_frequency:           float = 0.0
    raw_duty_cycle:          float = 0.0
    raw_stability:           float = 0.0

    @classmethod
    def null(cls) -> "RewardDecomposition":
        return cls(total=0.0)

    @classmethod
    def invalid_circuit(cls, errors: list) -> "RewardDecomposition":
        return cls(total=-0.10, invalid_action_penalty=-0.10)

    @classmethod
    def convergence_fail(cls, w: RewardWeights) -> "RewardDecomposition":
        penalty = -w.convergence
        return cls(total=penalty, convergence_penalty=penalty)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_array(self) -> np.ndarray:
        """Return key scores as a float32 array for logging."""
        return np.array([
            self.total,
            self.waveform_similarity,
            self.amplitude_score,
            self.frequency_score,
            self.duty_cycle_score,
            self.stability_score,
            self.simulation_success_score,
            self.component_penalty,
            self.invalid_action_penalty,
            self.convergence_penalty,
            self.terminal_bonus,
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Core reward computation
# ---------------------------------------------------------------------------

@dataclass
class TaskTarget:
    """Target waveform specification used by the reward function."""
    frequency:    float    # Hz
    duty_cycle:   float    # [0, 1]
    amplitude:    float    # Vpp (volts)

    # Tolerances (relative for freq/amp, absolute for duty_cycle)
    freq_tol:  float = 0.10   # ±10 %
    dc_tol:    float = 0.05   # ±0.05
    amp_tol:   float = 0.10   # ±10 %

    # Optional rise/fall time constraints (seconds), None = unconstrained
    max_rise_time: float | None = None
    max_fall_time: float | None = None
    min_stability: float = 0.80

    def to_array(self) -> np.ndarray:
        return np.array([
            self.frequency, self.duty_cycle, self.amplitude,
            self.freq_tol, self.dc_tol, self.amp_tol,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _amplitude_score(measured_vpp: float, target_amp: float) -> float:
    """Score based on Vpp closeness. Linear decay, 0 at 2x error."""
    if target_amp <= 0:
        return 0.0
    rel_err = abs(measured_vpp - target_amp) / target_amp
    return float(np.clip(1.0 - rel_err, 0.0, 1.0))


def _frequency_score(measured_freq: float, target_freq: float) -> float:
    """Exponential score decaying with relative frequency error."""
    if target_freq <= 0 or measured_freq <= 0:
        return 0.0
    rel_err = abs(measured_freq - target_freq) / target_freq
    return exponential_score(rel_err, scale=5.0)


def _duty_cycle_score(measured_dc: float, target_dc: float) -> float:
    """Linear score for duty cycle accuracy."""
    abs_err = abs(measured_dc - target_dc)
    return float(np.clip(1.0 - 2.0 * abs_err, 0.0, 1.0))


def _edge_quality_score(rise_time: float, fall_time: float,
                         period: float,
                         max_rise: float | None,
                         max_fall: float | None) -> float:
    """
    Score for edge sharpness (rise/fall time relative to period).
    A good square wave has rise/fall < 10% of the period.
    """
    if period <= 0:
        return 0.5  # neutral if period unknown

    scores = []
    if np.isfinite(rise_time) and rise_time > 0:
        rise_frac = rise_time / period
        s = float(np.clip(1.0 - 10.0 * rise_frac, 0.0, 1.0))
        if max_rise is not None:
            s = min(s, 1.0 if rise_time <= max_rise else 0.0)
        scores.append(s)

    if np.isfinite(fall_time) and fall_time > 0:
        fall_frac = fall_time / period
        s = float(np.clip(1.0 - 10.0 * fall_frac, 0.0, 1.0))
        if max_fall is not None:
            s = min(s, 1.0 if fall_time <= max_fall else 0.0)
        scores.append(s)

    return float(np.mean(scores)) if scores else 0.5


def _check_all_thresholds(
    metrics: WaveformMetrics,
    target: TaskTarget,
) -> bool:
    """Return True iff every waveform constraint is simultaneously satisfied."""
    if not metrics.sim_success:
        return False
    freq_ok = abs(metrics.frequency - target.frequency) / max(target.frequency, 1e-9) <= target.freq_tol
    dc_ok   = abs(metrics.duty_cycle - target.duty_cycle) <= target.dc_tol
    amp_ok  = abs(metrics.vpp - target.amplitude) / max(target.amplitude, 1e-9) <= target.amp_tol
    stab_ok = metrics.stability >= target.min_stability
    return bool(freq_ok and dc_ok and amp_ok and stab_ok)


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    metrics:                WaveformMetrics,
    target:                 TaskTarget,
    sim_success:            bool,
    n_components:           int,
    max_components:         int,
    n_invalid_actions:      int,
    max_steps:              int,
    n_convergence_failures: int,
    weights:                RewardWeights,
    over_budget:            bool,
    time_array:             "np.ndarray | None" = None,
    voltage_array:          "np.ndarray | None" = None,
) -> RewardDecomposition:
    """
    Compute the full decomposed reward for one episode finalisation.

    Parameters
    ----------
    metrics    : WaveformMetrics from the last simulation
    target     : TaskTarget specification
    sim_success: did the SPICE sim complete successfully?
    n_components, max_components: circuit size info
    n_invalid_actions, max_steps: action quality info
    n_convergence_failures: total convergence failures this episode
    weights    : RewardWeights for this task
    over_budget: True if n_components > task budget
    time_array, voltage_array: raw waveform for similarity computation

    Returns
    -------
    RewardDecomposition
    """
    w = weights

    # --- Simulation success bonus ---
    sim_score = 1.0 if sim_success else 0.0

    if not sim_success:
        conv_penalty = float(np.clip(n_convergence_failures * 0.1, 0.0, 1.0))
        invalid_penalty = float(np.clip(n_invalid_actions / max(max_steps, 1), 0.0, 1.0))
        total = (
            w.simulation_success * sim_score
            - w.convergence * conv_penalty
            - w.invalid_action * invalid_penalty
        )
        return RewardDecomposition(
            total=float(np.clip(total, -1.0, 1.0)),
            simulation_success_score=w.simulation_success * sim_score,
            convergence_penalty=-w.convergence * conv_penalty,
            invalid_action_penalty=-w.invalid_action * invalid_penalty,
        )

    # --- Compute sub-scores ---

    # Waveform similarity
    if time_array is not None and voltage_array is not None:
        raw_wsim = compute_waveform_similarity(
            time_array, voltage_array,
            target.frequency, target.duty_cycle, target.amplitude
        )
    else:
        # Approximate from metrics if raw arrays unavailable
        raw_wsim = 0.0
        if metrics.frequency > 0 and metrics.vpp > 0:
            f_s = _frequency_score(metrics.frequency, target.frequency)
            d_s = _duty_cycle_score(metrics.duty_cycle, target.duty_cycle)
            a_s = _amplitude_score(metrics.vpp, target.amplitude)
            raw_wsim = (f_s + d_s + a_s) / 3.0

    raw_amp  = _amplitude_score(metrics.vpp, target.amplitude)
    raw_freq = _frequency_score(metrics.frequency, target.frequency)
    raw_dc   = _duty_cycle_score(metrics.duty_cycle, target.duty_cycle)
    raw_stab = metrics.stability

    # Edge quality folds into stability score (average if both finite)
    period = 1.0 / max(metrics.frequency, 1e-10)
    edge_q = _edge_quality_score(
        metrics.rise_time, metrics.fall_time, period,
        target.max_rise_time, target.max_fall_time,
    )
    blended_stability = float(np.mean([raw_stab, edge_q]))

    # --- Penalties ---
    comp_pen    = float(np.clip(n_components / max(max_components, 1), 0.0, 1.0))
    invalid_pen = float(np.clip(n_invalid_actions / max(max_steps, 1), 0.0, 1.0))
    conv_pen    = float(np.clip(n_convergence_failures * 0.1, 0.0, 1.0))
    budget_pen  = 1.0 if over_budget else 0.0

    # --- Weighted sum ---
    pos = (
        w.waveform_similarity * raw_wsim
        + w.amplitude          * raw_amp
        + w.frequency          * raw_freq
        + w.duty_cycle         * raw_dc
        + w.stability          * blended_stability
        + w.simulation_success * sim_score
    )
    neg = (
        w.component_penalty * comp_pen
        + w.invalid_action  * invalid_pen
        + w.convergence     * conv_pen
        + w.overbudget      * budget_pen
    )

    # --- Terminal bonus ---
    all_met = _check_all_thresholds(metrics, target)
    bonus   = w.terminal_bonus if all_met else 0.0

    total = float(np.clip(pos - neg + bonus, -1.0, 1.0))

    return RewardDecomposition(
        total=total,
        waveform_similarity=w.waveform_similarity * raw_wsim,
        amplitude_score=w.amplitude * raw_amp,
        frequency_score=w.frequency * raw_freq,
        duty_cycle_score=w.duty_cycle * raw_dc,
        stability_score=w.stability * blended_stability,
        simulation_success_score=w.simulation_success * sim_score,
        component_penalty=-w.component_penalty * comp_pen,
        invalid_action_penalty=-w.invalid_action * invalid_pen,
        convergence_penalty=-w.convergence * conv_pen,
        overbudget_penalty=-w.overbudget * budget_pen,
        terminal_bonus=bonus,
        all_thresholds_met=all_met,
        raw_waveform_sim=raw_wsim,
        raw_amplitude=raw_amp,
        raw_frequency=raw_freq,
        raw_duty_cycle=raw_dc,
        raw_stability=blended_stability,
    )
