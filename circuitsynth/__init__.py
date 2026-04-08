"""
CircuitSynth: An OpenEnv-compliant RL environment for electronic circuit synthesis.

The agent incrementally constructs SPICE netlists (transistor-based astable
oscillator circuits) and receives reward based on how closely the simulated
output waveform matches a target square wave.

API:
    from circuitsynth import CircuitSynthEnv

    env = CircuitSynthEnv(task_id="squarewave-easy", seed=42, mock_sim=False)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    full_state = env.state()
"""

from .env import CircuitSynthEnv
from .tasks import TASK_REGISTRY, TaskSpec
from .action_space import Action, ActionType
from .components import ComponentType, COMPONENT_LIBRARY, NODE_NAMES, N_NODES
from .waveform import WaveformMetrics
from .reward import RewardDecomposition, RewardWeights
from .netlist import Netlist

__version__ = "1.0.0"
__all__ = [
    "CircuitSynthEnv",
    "TASK_REGISTRY",
    "TaskSpec",
    "Action",
    "ActionType",
    "ComponentType",
    "COMPONENT_LIBRARY",
    "NODE_NAMES",
    "N_NODES",
    "WaveformMetrics",
    "RewardDecomposition",
    "RewardWeights",
    "Netlist",
]
