"""
Observation builder for CircuitSynth.

The flat RL observation vector has shape (OBSERVATION_DIM,) = (269,) float32,
composed of:

  [0   : 144]  adjacency_matrix (12×12, flattened)
  [144 : 240]  component_features (12 components × 8 features, flattened)
  [240 : 250]  waveform_metrics (10 scalar metrics, normalised)
  [250 : 256]  task_target (6 values: freq, dc, amp, freq_tol, dc_tol, amp_tol)
  [256 : 260]  flags (4 bits: has_finalized, is_invalid, conv_failed, terminated)
  [260 : 263]  budget (n_components/max, steps_remaining/max_steps, n_invalid/max_steps)
  [263 : 269]  reward_history (last 6 reward totals, normalised to [-1,1])

Total: 269 floats.

A graph-structured (dict) observation is also available via build_graph_obs().
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .components import MAX_COMPONENTS, N_NODES
from .netlist import Netlist
from .waveform import WaveformMetrics
from .tasks import TaskSpec
from .reward import RewardDecomposition


# ---------------------------------------------------------------------------
# Observation dimensions
# ---------------------------------------------------------------------------

OBS_ADJ_FLAT     = N_NODES * N_NODES          # 144
OBS_COMP_FLAT    = MAX_COMPONENTS * 8          # 96
OBS_WAVEFORM     = 10
OBS_TARGET       = 6
OBS_FLAGS        = 4
OBS_BUDGET       = 3
OBS_REWARD_HIST  = 6

OBSERVATION_DIM = (
    OBS_ADJ_FLAT + OBS_COMP_FLAT + OBS_WAVEFORM
    + OBS_TARGET + OBS_FLAGS + OBS_BUDGET + OBS_REWARD_HIST
)   # = 269


# ---------------------------------------------------------------------------
# Normalisation helpers for waveform metrics
# ---------------------------------------------------------------------------

_METRIC_SCALES = np.array([
    1e4,    # frequency  → divide by 10 kHz
    1.0,    # duty_cycle (already [0,1])
    20.0,   # voh        → divide by 20 V
    20.0,   # vol        → divide by 20 V
    20.0,   # vpp        → divide by 20 V
    1e-3,   # rise_time  → divide by 1 ms
    1e-3,   # fall_time  → divide by 1 ms
    1e-2,   # settling   → divide by 10 ms
    1.0,    # stability  (already [0,1])
    50.0,   # n_cycles   → divide by 50
], dtype=np.float64)


def _normalise_metrics(arr: np.ndarray) -> np.ndarray:
    """Clip-normalise waveform metric array to roughly [0, 1]."""
    normed = np.clip(arr / _METRIC_SCALES, 0.0, 3.0).astype(np.float32)
    return normed


# ---------------------------------------------------------------------------
# Main observation builder
# ---------------------------------------------------------------------------

def build_observation(
    netlist: Netlist,
    task: TaskSpec,
    step_count: int,
    metrics: WaveformMetrics,
    has_finalized: bool,
    n_invalid_actions: int,
    convergence_failed: bool,
    terminated: bool = False,
    reward_history: "list[float] | None" = None,
) -> np.ndarray:
    """
    Build the flat float32 observation vector.

    Parameters
    ----------
    netlist           : current placed circuit
    task              : active TaskSpec
    step_count        : number of actions taken so far
    metrics           : WaveformMetrics from last simulation (or .null())
    has_finalized     : whether the agent has submitted the circuit
    n_invalid_actions : cumulative illegal-action count
    convergence_failed: True if the last sim did not converge
    terminated        : episode is done
    reward_history    : list of last ≤6 total rewards

    Returns
    -------
    np.ndarray of shape (OBSERVATION_DIM,)
    """
    # -- adjacency matrix ---
    adj = netlist.to_adjacency_matrix(N_NODES).ravel()            # (144,)

    # -- component features ---
    comp_feat = netlist.get_component_features(MAX_COMPONENTS).ravel()  # (96,)

    # -- waveform metrics ---
    raw_metrics = metrics.to_array()                              # (10,) float32
    norm_metrics = _normalise_metrics(raw_metrics.astype(np.float64))

    # -- task target ---
    target_arr = task.target.to_array()                           # (6,) float32
    # Normalise: freq→/10kHz, dc unchanged, amp→/20V, tols unchanged
    target_arr_norm = target_arr.copy()
    target_arr_norm[0] /= 1e4   # freq
    target_arr_norm[2] /= 20.0  # amplitude
    target_arr_norm = np.clip(target_arr_norm, 0.0, 3.0)

    # -- flags ---
    flags = np.array([
        float(has_finalized),
        float(n_invalid_actions > 0),
        float(convergence_failed),
        float(terminated),
    ], dtype=np.float32)                                          # (4,)

    # -- budget ---
    budget = np.array([
        len(netlist) / max(task.max_components, 1),
        max(task.max_steps - step_count, 0) / max(task.max_steps, 1),
        min(n_invalid_actions / max(task.max_steps, 1), 1.0),
    ], dtype=np.float32)                                          # (3,)

    # -- reward history ---
    hist = reward_history or []
    hist_arr = np.zeros(OBS_REWARD_HIST, dtype=np.float32)
    for i, r in enumerate(hist[-OBS_REWARD_HIST:]):
        hist_arr[i] = float(np.clip(r, -1.0, 1.0))

    obs = np.concatenate([
        adj, comp_feat, norm_metrics, target_arr_norm,
        flags, budget, hist_arr,
    ]).astype(np.float32)

    assert obs.shape == (OBSERVATION_DIM,), f"Obs shape mismatch: {obs.shape}"
    return obs


# ---------------------------------------------------------------------------
# Graph-structured observation (for GNN / graph-based policies)
# ---------------------------------------------------------------------------

def build_graph_obs(
    netlist: Netlist,
    task: TaskSpec,
    step_count: int,
    metrics: WaveformMetrics,
    has_finalized: bool,
    n_invalid_actions: int,
    convergence_failed: bool,
) -> Dict[str, Any]:
    """
    Return a graph-structured observation dict compatible with PyG / DGL.

    Schema
    ------
    {
      "node_features"  : np.ndarray (N_NODES, 3)   — [is_vcc, is_gnd, degree]
      "edge_index"     : np.ndarray (2, E)          — adjacency edges
      "component_nodes": np.ndarray (n_comps, 8)   — per-component features
      "global_context" : np.ndarray (29,)           — waveform + target + flags
    }
    """
    adj = netlist.to_adjacency_matrix(N_NODES)

    # Node features
    degree = adj.sum(axis=1)
    is_vcc = np.zeros(N_NODES, dtype=np.float32); is_vcc[0] = 1.0
    is_gnd = np.zeros(N_NODES, dtype=np.float32); is_gnd[1] = 1.0
    node_feat = np.stack([is_vcc, is_gnd, degree / max(degree.max(), 1.0)], axis=1)

    # Edge index (non-zero entries)
    src, dst = np.where(adj > 0)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    # Component nodes
    comp_feat = netlist.get_component_features(MAX_COMPONENTS)[:len(netlist)]

    # Global context
    norm_metrics = _normalise_metrics(metrics.to_array().astype(np.float64))
    target_arr = task.target.to_array()
    target_arr[0] /= 1e4; target_arr[2] /= 20.0
    flags = np.array([
        float(has_finalized), float(n_invalid_actions > 0),
        float(convergence_failed),
        len(netlist) / max(task.max_components, 1),
        max(task.max_steps - step_count, 0) / max(task.max_steps, 1),
    ], dtype=np.float32)
    global_ctx = np.concatenate([norm_metrics, target_arr, flags]).astype(np.float32)

    return {
        "node_features":   node_feat,
        "edge_index":      edge_index,
        "component_nodes": comp_feat,
        "global_context":  global_ctx,
    }
