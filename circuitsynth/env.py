"""
CircuitSynthEnv — OpenEnv-compliant RL environment for circuit synthesis.

The agent builds a transistor-based astable oscillator circuit step-by-step
by selecting typed actions from a structured space. When the agent submits
(FINALIZE action), ngspice runs a transient simulation and the resulting
waveform is compared to a target square wave. A decomposed shaped reward
in [-1, 1] is returned.

API (OpenEnv + Gymnasium compatible):
    env = CircuitSynthEnv(task_id="squarewave-easy", seed=42)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    full_state = env.state()

Gymnasium spaces:
    observation_space : Box(shape=(269,), dtype=float32)
    action_space      : MultiDiscrete([4, 7, 20, 12, 12, 12, 12])

MDP properties:
    - Episodic task with a fixed step budget per task
    - Deterministic under a fixed seed (same seed → same episode)
    - Partial observability: agent sees circuit graph + sim feedback,
      not the internal circuit physics
    - Invalid actions penalised but do NOT crash the environment
    - Non-convergent SPICE runs penalised with a diagnostic info object
    - Terminal: after FINALIZE or max_steps reached (truncation)
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, MultiDiscrete
except ImportError:
    import gym
    from gym.spaces import Box, MultiDiscrete

from .action_space import (
    ACTION_NVEC,
    Action,
    ActionMask,
    ActionType,
    build_action_mask,
)
from .components import (
    COMPONENT_LIBRARY,
    NODE_NAMES,
    N_NODES,
    ComponentType,
)
from .netlist import ActionResult, Netlist
from .observation import OBSERVATION_DIM, build_observation, build_graph_obs
from .reward import RewardDecomposition, compute_reward
from .simulator import NgSpiceSimulator, SimResult
from .tasks import TASK_REGISTRY, TaskSpec, get_task
from .utils import get_logger, seed_everything
from .waveform import WaveformMetrics, analyze_waveform

logger = get_logger("circuitsynth.env")


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class CircuitSynthEnv(gym.Env):
    """
    OpenEnv / Gymnasium RL environment for square-wave circuit synthesis.

    Parameters
    ----------
    task_id : str
        One of "squarewave-easy", "squarewave-medium", "squarewave-hard".
    seed : int
        Global RNG seed — makes episodes fully reproducible.
    mock_sim : bool
        If True, use the built-in mock waveform simulator instead of ngspice.
        Recommended during development or in environments without ngspice.
    ngspice_bin : str
        Path to the ngspice executable (default: "ngspice" from PATH).
    render_mode : str | None
        Unused; included for Gymnasium API compatibility.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # ------------------------------------------------------------------
    def __init__(
        self,
        task_id:     str  = "squarewave-easy",
        seed:        int  = 42,
        mock_sim:    bool = False,
        ngspice_bin: str  = "ngspice",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.task_id    = task_id
        self.seed       = seed
        self.mock_sim   = mock_sim
        self.render_mode = render_mode

        # Load task
        self.task: TaskSpec = get_task(task_id)

        # Create SPICE simulator
        self.simulator = NgSpiceSimulator(
            mock=mock_sim, ngspice_bin=ngspice_bin, timeout=30.0
        )

        # Gymnasium spaces
        self.action_space = MultiDiscrete(ACTION_NVEC)
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(OBSERVATION_DIM,), dtype=np.float32
        )

        # Episode state (initialised in reset)
        self._rng:                  Optional[np.random.Generator] = None
        self._netlist:              Netlist = Netlist()
        self._step_count:           int = 0
        self._n_invalid_actions:    int = 0
        self._n_convergence_fail:   int = 0
        self._has_finalized:        bool = False
        self._terminated:           bool = False
        self._truncated:            bool = False
        self._last_sim:             Optional[SimResult] = None
        self._last_metrics:         WaveformMetrics = WaveformMetrics.null()
        self._last_reward_decomp:   RewardDecomposition = RewardDecomposition.null()
        self._reward_history:       deque = deque(maxlen=6)
        self._episode_count:        int = 0
        self._episode_invalid_acts: list = []   # per-episode invalid-action counts
        self._episode_returns:      list = []   # total rewards per episode

    # ------------------------------------------------------------------
    # OpenEnv / Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        seed : int | None
            Override the global seed for this episode.
        task_id : str | None
            Switch to a different task.
        options : dict | None
            Unused; Gymnasium API compatibility.

        Returns
        -------
        obs  : np.ndarray  — initial observation (all zeros)
        info : dict        — episode metadata
        """
        # Optionally switch task
        if task_id is not None and task_id != self.task_id:
            self.task_id = task_id
            self.task = get_task(task_id)
            logger.info("Switched to task: %s", task_id)

        # Seed RNG
        effective_seed = seed if seed is not None else self.seed
        self._rng = seed_everything(effective_seed)
        self.simulator.set_rng(self._rng)
        super().reset(seed=effective_seed)

        # Record last episode stats before reset
        if self._episode_count > 0:
            self._episode_invalid_acts.append(self._n_invalid_actions)

        # Reset episode state
        self._netlist             = Netlist()
        self._step_count          = 0
        self._n_invalid_actions   = 0
        self._n_convergence_fail  = 0
        self._has_finalized       = False
        self._terminated          = False
        self._truncated           = False
        self._last_sim            = None
        self._last_metrics        = WaveformMetrics.null()
        self._last_reward_decomp  = RewardDecomposition.null()
        self._reward_history.clear()
        self._episode_count      += 1

        obs  = self._build_obs()
        info = self._build_info(step_reward=None, invalid=False)
        return obs, info

    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply one action to the environment.

        Parameters
        ----------
        action : np.ndarray of shape (7,) with integer values

        Returns
        -------
        obs        : np.ndarray
        reward     : float
        terminated : bool — episode ended by FINALIZE or invalid terminal
        truncated  : bool — episode ended by max_steps
        info       : dict — diagnostics including reward decomposition
        """
        assert not (self._terminated or self._truncated), (
            "Episode is done. Call reset() before stepping."
        )

        typed = Action.decode(np.asarray(action, dtype=np.int64))
        reward   = 0.0
        invalid  = False
        step_decomp = RewardDecomposition.null()

        # ---- Dispatch action ----
        if typed.action_type == ActionType.ADD_COMPONENT:
            result, tiny_reward = self._apply_add(typed)
            invalid = not result.success
            reward  = tiny_reward

        elif typed.action_type == ActionType.REMOVE_COMPONENT:
            result = self._netlist.remove_component_by_index(typed.remove_idx)
            invalid = not result.success
            if invalid:
                reward = -0.01
                self._n_invalid_actions += 1

        elif typed.action_type == ActionType.FINALIZE:
            reward, step_decomp = self._finalize()
            self._last_reward_decomp = step_decomp
            self._terminated = True

        elif typed.action_type == ActionType.NO_OP:
            reward = -0.002  # tiny step penalty to discourage loitering

        self._step_count += 1
        self._reward_history.append(reward)

        # ---- Check truncation ----
        if self._step_count >= self.task.max_steps and not self._terminated:
            # Auto-finalize on truncation with penalty
            if not self._has_finalized:
                trunc_reward, trunc_decomp = self._finalize()
                reward += trunc_reward
                self._last_reward_decomp = trunc_decomp
            self._truncated = True

        obs  = self._build_obs()
        info = self._build_info(step_reward=reward, invalid=invalid)

        if self._terminated or self._truncated:
            self._episode_returns.append(reward)

        return obs, float(reward), self._terminated, self._truncated, info

    # ------------------------------------------------------------------

    def state(self) -> dict:
        """
        Return the full MDP state dict (includes info hidden from the policy).

        Useful for offline analysis, curriculum learning, and debugging.
        """
        return {
            "episode":             self._episode_count,
            "step":                self._step_count,
            "task":                self.task.to_dict(),
            "netlist":             self._netlist.to_dict(),
            "has_finalized":       self._has_finalized,
            "terminated":         self._terminated,
            "truncated":           self._truncated,
            "n_invalid_actions":   self._n_invalid_actions,
            "n_convergence_fail":  self._n_convergence_fail,
            "budget_used":         len(self._netlist),
            "budget_remaining":    self.task.max_components - len(self._netlist),
            "steps_remaining":     self.task.max_steps - self._step_count,
            "waveform_metrics":    self._last_metrics.to_dict(),
            "reward_decomposition": self._last_reward_decomp.to_dict(),
            "last_sim": (
                self._last_sim.to_dict() if self._last_sim else None
            ),
            "action_mask":         self._get_action_mask().to_flat_mask().tolist(),
        }

    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        """Minimal text rendering of the current circuit state."""
        lines = [
            f"=== CircuitSynth [{self.task_id}] step {self._step_count} ===",
            f"Components ({len(self._netlist)}/{self.task.max_components}):",
        ]
        for pc in self._netlist.components:
            conn = {t: NODE_NAMES[n] for t, n in pc.connections.items()}
            lines.append(f"  {pc.comp_id:6s} {pc.comp_type.name:12s} {pc.value:.3e}  {conn}")
        if self._last_metrics.sim_success:
            m = self._last_metrics
            lines.append(
                f"Last sim: freq={m.frequency:.1f} Hz  dc={m.duty_cycle:.3f}  "
                f"Vpp={m.vpp:.2f} V  stab={m.stability:.3f}"
            )
        if mode == "human":
            print("\n".join(lines))
        return "\n".join(lines)

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal implementation helpers
    # ------------------------------------------------------------------

    def _apply_add(self, action: Action) -> Tuple[ActionResult, float]:
        """Validate and apply an ADD_COMPONENT action. Returns (result, reward)."""
        spec = COMPONENT_LIBRARY.get(action.component_type)
        if spec is None or action.component_type == ComponentType.NONE:
            self._n_invalid_actions += 1
            return ActionResult(success=False, error="Invalid component type"), -0.02

        value = spec.get_value(action.value_idx)

        # Build connection dict based on terminal count
        if action.component_type == ComponentType.NPN_BJT:
            connections = {
                "C": action.node_a,
                "B": action.node_b,
                "E": action.node_c,
            }
        else:
            terminals = spec.terminal_names
            connections = {
                terminals[0]: action.node_a,
                terminals[1]: action.node_b,
            }

        result = self._netlist.add_component(
            comp_type=action.component_type,
            value=value,
            value_idx=action.value_idx,
            connections=connections,
        )

        if not result.success:
            self._n_invalid_actions += 1
            return result, -0.02

        # Small positive reward for each valid component placed
        placement_reward = 0.005
        return result, placement_reward

    # ------------------------------------------------------------------

    def _finalize(self) -> Tuple[float, RewardDecomposition]:
        """
        Submit the circuit for simulation and compute the full reward.

        Returns (total_reward, RewardDecomposition).
        """
        self._has_finalized = True

        # Validate
        val = self._netlist.validate()
        if not val.valid:
            logger.debug("Circuit invalid: %s", val.errors)
            self._n_invalid_actions += 1
            decomp = RewardDecomposition.invalid_circuit(val.errors)
            return decomp.total, decomp

        # Generate SPICE netlist string
        netlist_str = self._netlist.to_spice(
            output_node=self.task.output_node,
            stop_time=self.task.stop_time,
            step_size=self.task.step_size,
        )

        logger.debug("Running SPICE simulation (%s)…", self.task_id)

        # Run simulation
        sim_result = self.simulator.run_transient(
            netlist_str=netlist_str,
            stop_time=self.task.stop_time,
            step_size=self.task.step_size,
            netlist_dict=self._netlist.to_dict(),
            target_freq=self.task.target.frequency,
        )
        self._last_sim = sim_result

        if not sim_result.success:
            self._n_convergence_fail += 1
            logger.debug("Sim failed: %s", sim_result.error_msg)

        # Extract waveform metrics
        if sim_result.success and sim_result.time_array is not None:
            self._last_metrics = analyze_waveform(
                time_array=sim_result.time_array,
                voltage_array=sim_result.voltage_array,
                target_frequency=self.task.target.frequency,
            )
        else:
            self._last_metrics = WaveformMetrics.null()

        # Compute reward
        decomp = compute_reward(
            metrics=self._last_metrics,
            target=self.task.target,
            sim_success=sim_result.success,
            n_components=len(self._netlist),
            max_components=self.task.max_components,
            n_invalid_actions=self._n_invalid_actions,
            max_steps=self.task.max_steps,
            n_convergence_failures=self._n_convergence_fail,
            weights=self.task.reward_weights,
            over_budget=len(self._netlist) > self.task.max_components,
            time_array=sim_result.time_array,
            voltage_array=sim_result.voltage_array,
        )

        logger.debug(
            "Reward: %.4f  (freq=%.1f Hz target=%.1f Hz  "
            "dc=%.3f target=%.3f  Vpp=%.2f target=%.2f)",
            decomp.total,
            self._last_metrics.frequency, self.task.target.frequency,
            self._last_metrics.duty_cycle, self.task.target.duty_cycle,
            self._last_metrics.vpp, self.task.target.amplitude,
        )

        return decomp.total, decomp

    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        return build_observation(
            netlist=self._netlist,
            task=self.task,
            step_count=self._step_count,
            metrics=self._last_metrics,
            has_finalized=self._has_finalized,
            n_invalid_actions=self._n_invalid_actions,
            convergence_failed=self._n_convergence_fail > 0,
            terminated=self._terminated or self._truncated,
            reward_history=list(self._reward_history),
        )

    def _build_info(self, step_reward: Optional[float], invalid: bool) -> dict:
        m = self._last_metrics
        d = self._last_reward_decomp
        return {
            "task_id":              self.task_id,
            "step":                 self._step_count,
            "n_components":         len(self._netlist),
            "budget_remaining":     self.task.max_components - len(self._netlist),
            "steps_remaining":      self.task.max_steps - self._step_count,
            "invalid_action":       invalid,
            "n_invalid_actions":    self._n_invalid_actions,
            "n_convergence_fail":   self._n_convergence_fail,
            "step_reward":          step_reward,
            "reward_decomposition": d.to_dict(),
            "waveform_metrics":     m.to_dict(),
            "all_thresholds_met":   d.all_thresholds_met,
            "sim_success":          (self._last_sim.success
                                     if self._last_sim else None),
            "action_mask":          self._get_action_mask().to_flat_mask().tolist(),
        }

    def _get_action_mask(self) -> ActionMask:
        return build_action_mask(
            n_placed=len(self._netlist),
            max_components=self.task.max_components,
        )

    # ------------------------------------------------------------------
    # Graph observation interface (for GNN-based policies)
    # ------------------------------------------------------------------

    def graph_observation(self) -> dict:
        """Return a graph-structured observation for GNN / graph policies."""
        return build_graph_obs(
            netlist=self._netlist,
            task=self.task,
            step_count=self._step_count,
            metrics=self._last_metrics,
            has_finalized=self._has_finalized,
            n_invalid_actions=self._n_invalid_actions,
            convergence_failed=self._n_convergence_fail > 0,
        )

    # ------------------------------------------------------------------
    # Evaluation / statistics
    # ------------------------------------------------------------------

    def get_episode_stats(self) -> dict:
        """Return aggregate statistics across all completed episodes."""
        returns = np.array(self._episode_returns) if self._episode_returns else np.array([0.0])
        inv_acts = np.array(self._episode_invalid_acts) if self._episode_invalid_acts else np.array([0.0])
        return {
            "n_episodes":          self._episode_count,
            "mean_return":         float(returns.mean()),
            "max_return":          float(returns.max()),
            "min_return":          float(returns.min()),
            "mean_invalid_acts":   float(inv_acts.mean()),
        }

    # ------------------------------------------------------------------
    # Convenience: scripted action interface (dict API)
    # ------------------------------------------------------------------

    def step_dict(self, action_dict: dict) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step using a human-friendly dict action.

        Example
        -------
        env.step_dict({
            "action_type": "ADD_COMPONENT",
            "component_type": "NPN_BJT",
            "value_idx": 0,
            "node_a": "N1",    # collector
            "node_b": "N3",    # base
            "node_c": "GND",   # emitter
        })
        """
        from .components import NODE_INDEX
        atype_map = {a.name: a for a in ActionType}
        ctype_map = {c.name: c for c in ComponentType}

        atype = atype_map.get(action_dict.get("action_type", "NO_OP"), ActionType.NO_OP)
        ctype = ctype_map.get(action_dict.get("component_type", "NONE"), ComponentType.NONE)

        def _node(key: str, default: int = 1) -> int:
            v = action_dict.get(key, NODE_NAMES[default])
            if isinstance(v, int):
                return int(np.clip(v, 0, N_NODES - 1))
            return NODE_INDEX.get(str(v).upper(), default)

        vec = np.array([
            int(atype),
            int(ctype),
            int(action_dict.get("value_idx", 0)),
            _node("node_a", 0),
            _node("node_b", 1),
            _node("node_c", 1),
            int(action_dict.get("remove_idx", 0)),
        ], dtype=np.int64)

        return self.step(vec)
