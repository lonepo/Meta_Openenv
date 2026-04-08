"""
Task registry for CircuitSynth-SquareWave.

Three tasks with increasing difficulty, all in the transistor-based
astable oscillator family:

  Task 1 — squarewave-easy   : 555 Hz, loose tolerances, 12-component budget
  Task 2 — squarewave-medium : 1 kHz,  tighter tolerances, 10-component budget
  Task 3 — squarewave-hard   : 2 kHz,  strict tolerances + edge quality, 8 components

Shared circuit family: NPN BJT cross-coupled (2-transistor) astable multivibrator
with RC timing networks. Reference topology:

    VCC ─── R1 ─── C ─── B of Q2
    VCC ─── R3 ─── C of Q1 ─── (output N1)
    Q1 C ─── R2_b ─── B of Q1 ─── C2 ─── C of Q2
    Q1, Q2 emitters → GND
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .reward import RewardWeights, TaskTarget


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """Full specification for one CircuitSynth task."""
    task_id:         str
    description:     str
    target:          TaskTarget
    max_components:  int         # component budget
    max_steps:       int         # max episode length (actions)
    reward_weights:  RewardWeights
    output_node:     str         # SPICE node to probe (e.g. "N1")
    stop_time:       float       # transient sim stop time (s)
    step_size:       float       # TRAN step size (s)
    curriculum_hint: str         # natural-language hint for curriculum learning

    def to_dict(self) -> dict:
        return {
            "task_id":        self.task_id,
            "description":    self.description,
            "target":         self.target.to_dict(),
            "max_components": self.max_components,
            "max_steps":      self.max_steps,
            "output_node":    self.output_node,
            "stop_time":      self.stop_time,
            "step_size":      self.step_size,
        }


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskSpec] = {

    # -----------------------------------------------------------------------
    # EASY: 555 Hz square wave, loose tolerances, generous budget / time
    # -----------------------------------------------------------------------
    "squarewave-easy": TaskSpec(
        task_id="squarewave-easy",
        description=(
            "Generate a 555 Hz square wave with ~5 V amplitude and 50 % duty cycle. "
            "Loose tolerances (±20 % frequency, ±10 % amplitude, ±10 % duty cycle). "
            "12-component budget, 30 action steps."
        ),
        target=TaskTarget(
            frequency=555.0,
            duty_cycle=0.50,
            amplitude=5.0,
            freq_tol=0.20,   # ±20 %
            dc_tol=0.10,     # ±0.10 absolute
            amp_tol=0.20,    # ±20 %
            max_rise_time=None,
            max_fall_time=None,
            min_stability=0.70,
        ),
        max_components=12,
        max_steps=30,
        reward_weights=RewardWeights.easy(),
        output_node="N1",
        stop_time=20e-3,     # 20 ms — ≈11 cycles at 555 Hz
        step_size=5e-6,      # 5 µs time step
        curriculum_hint=(
            "Start by placing two NPN BJTs, two collector resistors to VCC, "
            "two base resistors, and two cross-coupling capacitors.  "
            "Connect emitters to GND and add a VCC supply node."
        ),
    ),

    # -----------------------------------------------------------------------
    # MEDIUM: 1 kHz square wave, tighter tolerances, reduced budget
    # -----------------------------------------------------------------------
    "squarewave-medium": TaskSpec(
        task_id="squarewave-medium",
        description=(
            "Generate a 1 kHz square wave with ~5 V amplitude and 50 % duty cycle. "
            "Tighter tolerances (±10 % frequency, ±5 % amplitude, ±5 % duty cycle). "
            "10-component budget, 25 action steps."
        ),
        target=TaskTarget(
            frequency=1000.0,
            duty_cycle=0.50,
            amplitude=5.0,
            freq_tol=0.10,   # ±10 %
            dc_tol=0.05,     # ±0.05 absolute
            amp_tol=0.10,    # ±10 %
            max_rise_time=100e-6,  # rise < 100 µs
            max_fall_time=100e-6,
            min_stability=0.85,
        ),
        max_components=10,
        max_steps=25,
        reward_weights=RewardWeights.medium(),
        output_node="N1",
        stop_time=10e-3,     # 10 ms — 10 cycles at 1 kHz
        step_size=2e-6,      # 2 µs time step
        curriculum_hint=(
            "Tune RC time constants: for 1 kHz use R ≈ 47 kΩ and C ≈ 15 nF "
            "(T ≈ 1.38·R·C per half-cycle in a symmetric astable)."
        ),
    ),

    # -----------------------------------------------------------------------
    # HARD: 2 kHz precision square wave, strict constraints, tight budget
    # -----------------------------------------------------------------------
    "squarewave-hard": TaskSpec(
        task_id="squarewave-hard",
        description=(
            "Generate a 2 kHz precision square wave with ~5 V amplitude and 50 % duty cycle. "
            "Strict tolerances (±5 % frequency, ±2 % amplitude, ±2 % duty cycle). "
            "Rise/fall time < 50 µs.  Stability > 0.95.  8-component budget, 20 steps."
        ),
        target=TaskTarget(
            frequency=2000.0,
            duty_cycle=0.50,
            amplitude=5.0,
            freq_tol=0.05,   # ±5 %
            dc_tol=0.02,     # ±0.02 absolute
            amp_tol=0.05,    # ±5 %
            max_rise_time=50e-6,   # rise < 50 µs
            max_fall_time=50e-6,
            min_stability=0.95,
        ),
        max_components=8,
        max_steps=20,
        reward_weights=RewardWeights.hard(),
        output_node="N1",
        stop_time=6e-3,      # 6 ms — 12 cycles at 2 kHz
        step_size=1e-6,      # 1 µs time step
        curriculum_hint=(
            "For 2 kHz: R ≈ 36 kΩ, C ≈ 10 nF.  Use minimal topology — exactly "
            "2 BJTs, 2 collector resistors, 2 cross-coupling capacitors, "
            "and a VCC supply (8 components total)."
        ),
    ),
}


def get_task(task_id: str) -> TaskSpec:
    """Retrieve a task by ID. Raises KeyError if not found."""
    if task_id not in TASK_REGISTRY:
        valid = list(TASK_REGISTRY.keys())
        raise KeyError(f"Unknown task '{task_id}'. Valid tasks: {valid}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> list:
    """Return a list of task info dicts."""
    return [spec.to_dict() for spec in TASK_REGISTRY.values()]
