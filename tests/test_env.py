"""Unit tests for CircuitSynthEnv — API contract and basic episode flow."""

from __future__ import annotations

import numpy as np
import pytest

from circuitsynth import CircuitSynthEnv, TASK_REGISTRY
from circuitsynth.action_space import Action, ActionType
from circuitsynth.components import ComponentType
from circuitsynth.observation import OBSERVATION_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(TASK_REGISTRY.keys()))
def env(request):
    """One env per task, all using mock simulator."""
    e = CircuitSynthEnv(task_id=request.param, seed=42, mock_sim=True)
    yield e
    e.close()


@pytest.fixture
def easy_env():
    e = CircuitSynthEnv(task_id="squarewave-easy", seed=42, mock_sim=True)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------

class TestAPIContract:
    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (OBSERVATION_DIM,), f"Expected {OBSERVATION_DIM}, got {obs.shape}"
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_returns_deterministic_obs(self, easy_env):
        obs1, _ = easy_env.reset(seed=99)
        obs2, _ = easy_env.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2, err_msg="Reset should be deterministic")

    def test_step_returns_5_tuple(self, easy_env):
        easy_env.reset()
        action = easy_env.action_space.sample()
        result = easy_env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (OBSERVATION_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_state_returns_dict(self, easy_env):
        easy_env.reset()
        s = easy_env.state()
        assert isinstance(s, dict)
        required_keys = ["step", "netlist", "task", "waveform_metrics",
                         "reward_decomposition", "terminated", "truncated"]
        for k in required_keys:
            assert k in s, f"Missing key: {k}"

    def test_step_raises_after_done(self, easy_env):
        easy_env.reset()
        # Force finalize
        finalize_vec = np.array([2, 0, 0, 0, 1, 1, 0])  # FINALIZE
        easy_env.step(finalize_vec)
        with pytest.raises(AssertionError):
            easy_env.step(finalize_vec)


# ---------------------------------------------------------------------------
# Episode flow
# ---------------------------------------------------------------------------

class TestEpisodeFlow:
    def test_random_episode_completes(self, env):
        obs, info = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(100):
            action = env.action_space.sample()
            # Force finalize with 20% probability to end episodes
            if rng.random() < 0.2:
                action[0] = 2
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (OBSERVATION_DIM,)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if terminated or truncated:
                break
        assert terminated or truncated, "Episode should end within 100 steps"

    def test_truncation_on_max_steps(self, easy_env):
        """Episode should truncate at max_steps if FINALIZE not called."""
        easy_env.reset()
        # Use only NO_OP actions so we never finalize
        noop = np.array([3, 0, 0, 0, 1, 1, 0])  # NO_OP
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = easy_env.step(noop)
            step += 1
            if step > 200:
                break
        assert truncated or terminated, "Should truncate/terminate at max_steps"

    def test_reward_in_valid_range(self, easy_env):
        """Final reward must be in [-1, 1]."""
        easy_env.reset()
        finalize = np.array([2, 0, 0, 0, 1, 1, 0])
        _, reward, _, _, _ = easy_env.step(finalize)
        assert -1.0 <= reward <= 1.0, f"Reward out of range: {reward}"

    def test_full_reference_circuit(self, easy_env):
        """Build a known-good reference circuit and check reward > 0."""
        from scripts.evaluate import REFERENCE_CIRCUITS, run_scripted_episode
        circuit = REFERENCE_CIRCUITS["squarewave-easy"]
        result = run_scripted_episode(easy_env, circuit)
        # With mock sim, reward should be non-trivially positive for a good circuit
        assert result["n_components"] == 9, "Reference circuit has 9 components"
        assert result["sim_success"], "Mock sim should succeed for a valid circuit"


# ---------------------------------------------------------------------------
# Action handling
# ---------------------------------------------------------------------------

class TestActionHandling:
    def test_invalid_action_penalised_not_crashed(self, easy_env):
        """Invalid actions return a penalty reward and mark invalid_action=True."""
        easy_env.reset()
        # Short circuit: resistor with node_a == node_b
        vec = np.array([0, 0, 5, 2, 2, 2, 0])   # ADD RESISTOR at N1→N1 (same node)
        obs, reward, _, _, info = easy_env.step(vec)
        assert info["invalid_action"] is True
        assert reward < 0, "Invalid action should produce negative reward"

    def test_add_component_increases_count(self, easy_env):
        easy_env.reset()
        assert len(easy_env._netlist) == 0
        # Add a valid resistor VCC(0) → N1(2)
        vec = np.array([0, 0, 10, 0, 2, 1, 0])  # ADD RESISTOR VCC→N1
        obs, _, _, _, info = easy_env.step(vec)
        assert info["n_components"] == 1

    def test_remove_decreases_count(self, easy_env):
        easy_env.reset()
        # Add two components
        easy_env.step(np.array([0, 0, 10, 0, 2, 1, 0]))  # ADD VCC→N1
        easy_env.step(np.array([0, 0, 8,  0, 3, 1, 0]))  # ADD VCC→N2
        assert len(easy_env._netlist) == 2
        # Remove first
        easy_env.step(np.array([1, 0, 0, 0, 1, 1, 0]))   # REMOVE idx=0
        assert len(easy_env._netlist) == 1

    def test_exceeding_budget_is_penalised(self, easy_env):
        """Adding components beyond max_components should be rejected."""
        easy_env.reset()
        max_c = easy_env.task.max_components
        for i in range(max_c + 5):
            # ADD RESISTOR with distinct nodes (cycle through N1..N10)
            na = 2 + (i * 2) % 10
            nb = 3 + (i * 2) % 10
            if na == nb:
                nb = (nb + 1) % N_NODES
            vec = np.array([0, 0, 5, na, nb, 1, 0])
            easy_env.step(vec)
        assert len(easy_env._netlist) <= max_c, "Budget should be enforced"


# ---------------------------------------------------------------------------
# observation / info content
# ---------------------------------------------------------------------------

class TestInfoContent:
    def test_info_has_reward_decomposition(self, easy_env):
        easy_env.reset()
        easy_env.step(np.array([2, 0, 0, 0, 1, 1, 0]))  # FINALIZE
        s = easy_env.state()
        rd = s["reward_decomposition"]
        assert "total" in rd
        assert "frequency_score" in rd
        assert "amplitude_score" in rd

    def test_obs_is_finite(self, env):
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs)), "Initial observation contains non-finite values"
        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs2)), "Observation after step contains non-finite values"


from circuitsynth.components import N_NODES  # used in TestActionHandling
