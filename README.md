---
title: CircuitSynth SquareWave RL Env
emoji: ⚡
colorFrom: gold
colorTo: black
sdk: docker
app_port: 7860
pinned: false
---

# CircuitSynth-SquareWave

> An OpenEnv-compliant RL environment for LLM-based electronic circuit synthesis using ngspice SPICE simulation.

The agent learns to design **transistor-based astable oscillator circuits** that generate target square waveforms by incrementally placing and connecting components from a fixed library.

---

## Overview

```
┌─────────────┐    action      ┌───────────────────┐   SPICE netlist  ┌──────────────┐
│  RL Policy  │ ──────────→   │  CircuitSynthEnv  │ ──────────────→  │   ngspice    │
│ (any algo)  │               │  (OpenEnv API)    │                  │  (simulator) │
└─────────────┘ ←──────────── └───────────────────┘ ←──────────────  └──────────────┘
                 obs + reward        circuit graph         waveform data
```

**Circuit family:** NPN BJT cross-coupled astable multivibrator (transistor-RC square-wave oscillator).

**Objective:** Train an RL agent to discover circuits that produce a target square wave with correct frequency, duty cycle, amplitude, and stability — using as few components as possible.

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/circuitsynth-squarewave
cd circuitsynth-squarewave
pip install -e .
```

For real SPICE simulation, install ngspice:
```bash
sudo apt-get install ngspice        # Ubuntu / Debian
brew install ngspice                # macOS (Homebrew)
```

Without ngspice, the environment uses a built-in **mock simulator** that estimates waveform quality from circuit topology. Set `mock_sim=True`.

### Basic Usage

```python
from circuitsynth import CircuitSynthEnv

# Create environment (mock mode — no ngspice required)
env = CircuitSynthEnv(task_id="squarewave-easy", seed=42, mock_sim=True)

# OpenEnv API
obs, info = env.reset()

for step in range(30):
    action = env.action_space.sample()          # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: reward={reward:.4f}  components={info['n_components']}")
    if terminated or truncated:
        break

# Full MDP state (for debugging / logging)
state = env.state()
print(state["reward_decomposition"])
```

### Scripted Human-Readable Actions

```python
# Use dict-style actions for scripting / debugging
obs, info = env.reset()

env.step_dict({"action_type": "ADD_COMPONENT", "component_type": "VSOURCE",
               "value_idx": 5, "node_a": "VCC", "node_b": "GND"})
env.step_dict({"action_type": "ADD_COMPONENT", "component_type": "RESISTOR",
               "value_idx": 5, "node_a": "VCC", "node_b": "N1"})
env.step_dict({"action_type": "ADD_COMPONENT", "component_type": "NPN_BJT",
               "value_idx": 0, "node_a": "N1", "node_b": "N3", "node_c": "GND"})
# ... build full circuit ...

obs, reward, terminated, truncated, info = env.step_dict({"action_type": "FINALIZE"})
print(f"Final reward: {reward:.4f}")
print(f"Frequency: {info['waveform_metrics']['frequency']:.1f} Hz")
```

---

## Tasks

| Task ID | Frequency | Duty Cycle | Amplitude | Budget | Steps | Difficulty |
|---------|-----------|------------|-----------|--------|-------|------------|
| `squarewave-easy`   | 555 Hz  | 50% | 5 V | 12 components | 30 | Easy   |
| `squarewave-medium` | 1000 Hz | 50% | 5 V | 10 components | 25 | Medium |
| `squarewave-hard`   | 2000 Hz | 50% | 5 V | 8 components  | 20 | Hard   |

### Task Tolerances

| Task | Freq tol | DC tol | Amp tol | Rise/Fall | Stability |
|------|----------|--------|---------|-----------|-----------|
| Easy   | ±20%   | ±10%   | ±20%   | —         | > 0.70    |
| Medium | ±10%   | ±5%    | ±10%   | < 100 µs  | > 0.85    |
| Hard   | ±5%    | ±2%    | ±5%    | < 50 µs   | > 0.95    |

---

## Action Space

**`gymnasium.spaces.MultiDiscrete([4, 7, 20, 12, 12, 12, 12])`** — 7 integers per action.

| Index | Dimension | Values | Description |
|-------|-----------|--------|-------------|
| 0 | `action_type`    | 0–3  | ADD / REMOVE / FINALIZE / NO_OP |
| 1 | `component_type` | 0–6  | RESISTOR / CAPACITOR / NPN_BJT / DIODE / VSOURCE / SWITCH / NONE |
| 2 | `value_idx`      | 0–19 | Log-spaced value (10 Ω–10 MΩ for R; 1 pF–1 mF for C; 1–15 V for VSource) |
| 3 | `node_a`         | 0–11 | First terminal: 0=VCC, 1=GND, 2..11=N1..N10 |
| 4 | `node_b`         | 0–11 | Second terminal |
| 5 | `node_c`         | 0–11 | Third terminal (BJT emitter) |
| 6 | `remove_idx`     | 0–11 | Index of component to remove |

**Invalid action masking** is supported via `info["action_mask"]` (flat boolean array of shape `(79,)`) compatible with `stable-baselines3[contrib]` `MaskablePPO`.

---

## Observation Space

**`gymnasium.spaces.Box(shape=(269,), dtype=float32)`**

| Slice | Size | Content |
|-------|------|---------|
| `[0:144]` | 144 | Adjacency matrix (12×12, normalised) |
| `[144:240]` | 96 | Component feature matrix (12 comps × 8 features) |
| `[240:250]` | 10 | Waveform metrics (freq, DC, VOH, VOL, Vpp, rise, fall, settle, stability, n_cycles) |
| `[250:256]` | 6 | Task target (freq, DC, amp, tolerances) |
| `[256:260]` | 4 | Flags (finalized, invalid, conv_failed, terminated) |
| `[260:263]` | 3 | Budget (components used/max, steps remaining, invalid rate) |
| `[263:269]` | 6 | Reward history (last 6 rewards) |

A **graph-structured observation** (for GNN policies) is available via `env.graph_observation()`.

---

## Reward Function

**Range:** `[0.0, 1.0]` (returned by the server and `inference.py`; internally shaped from `[-1, 1]`)

| Component | Weight (default) | Description |
|-----------|-----------------|-------------|
| `waveform_similarity` | 0.25 | Normalised MSE vs ideal square wave |
| `amplitude_score`     | 0.20 | Closeness of Vpp to target |
| `frequency_score`     | 0.20 | Exponential decay with relative freq error |
| `duty_cycle_score`    | 0.15 | Linear penalty on DC error |
| `stability_score`     | 0.10 | Period regularity + edge quality |
| `simulation_success`  | 0.05 | Binary bonus for SPICE convergence |
| `-component_penalty`  | 0.05 | n_components / max_components |
| `-invalid_action`     | 0.03 | Cumulative invalid action rate |
| `-convergence`        | 0.02 | Per-SPICE-failure penalty |
| `-overbudget`         | 0.02 | Binary if over component budget |
| `+terminal_bonus`     | 0.20 | All thresholds simultaneously met |

Full decomposition is logged in `info["reward_decomposition"]` at every step.

---

## Reference Circuit

The canonical 2-BJT astable multivibrator (9 components):

```
VCC ─┬── R1(1kΩ) ──┬── C1(15nF) ──┬── R4(47kΩ) ──┐
     │             N1              N4              │
     │              ├── Q1(C)     Q2(B) ──┐       │
     └── R2(1kΩ) ──┬── C2(15nF) ──┬── R3(47kΩ)  │
                   N2              N3              │
                    └── Q2(C)     Q1(B) ──┘       │
Q1.E, Q2.E → GND                                  │
Output at N1 (or N2)                               │
```

*f ≈ 1/(1.38·R·C) per half-cycle for symmetric circuit.*

Validate the reward function with:
```bash
python scripts/evaluate.py --reference --task squarewave-easy --mock
```

---

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from circuitsynth import CircuitSynthEnv

env = CircuitSynthEnv(task_id="squarewave-easy", mock_sim=True)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1,
)
model.learn(total_timesteps=500_000)
model.save("circuitsynth_ppo")

# Evaluate
obs, _ = env.reset()
for _ in range(30):
    action, _ = model.predict(obs)
    obs, r, done, trunc, info = env.step(action)
    if done or trunc:
        break
print(f"Reward: {r:.4f}  Freq: {info['waveform_metrics']['frequency']:.1f} Hz")
```

### Curriculum Learning

Start with Easy, then switch to harder tasks:
```python
for task_id in ["squarewave-easy", "squarewave-medium", "squarewave-hard"]:
    env = CircuitSynthEnv(task_id=task_id, mock_sim=False)
    model = PPO.load("circuitsynth_ppo")
    model.set_env(env)
    model.learn(total_timesteps=200_000)
    model.save(f"circuitsynth_ppo_{task_id}")
```

---

## Scripts

```bash
# Random baseline (no ngspice needed)
python scripts/baseline_inference.py --task squarewave-easy --episodes 20 --mock

# Reference circuit reward validation
python scripts/evaluate.py --reference --task squarewave-easy --mock

# Evaluate a trained SB3 model
python scripts/evaluate.py --model circuitsynth_ppo.zip --task squarewave-hard

# Run tests
pytest tests/ -v
```

---

## Project Structure

```
circuitsynth-squarewave/
├── README.md
├── Dockerfile              # HuggingFace Spaces compatible
├── openenv.yaml            # OpenEnv specification
├── requirements.txt
├── setup.py
│
├── circuitsynth/
│   ├── __init__.py         # Package API
│   ├── env.py              # CircuitSynthEnv (main class)
│   ├── tasks.py            # Task registry (easy / medium / hard)
│   ├── action_space.py     # Typed actions + MultiDiscrete encoding
│   ├── observation.py      # Observation builder (flat + graph)
│   ├── reward.py           # Decomposed reward function
│   ├── netlist.py          # Netlist graph + SPICE serializer + validation
│   ├── simulator.py        # ngspice subprocess wrapper + mock fallback
│   ├── waveform.py         # Waveform analysis (FFT, DTW, metrics)
│   ├── components.py       # Component library + SPICE models
│   └── utils.py            # Helpers: seeding, SI formatting, convergence detection
│
├── scripts/
│   ├── baseline_inference.py   # Reproducible random policy rollout
│   └── evaluate.py             # Offline evaluation + reference circuit
│
└── tests/
    ├── test_env.py
    ├── test_reward.py
    ├── test_simulator.py
    └── test_waveform.py
```

---

## Evaluation Metrics

The environment logs the following metrics per episode (in `info` dict and `env.state()`):

| Metric | Description |
|--------|-------------|
| `total_reward` | Final reward in [0, 1] (server) |
| `all_thresholds_met` | True if ALL waveform constraints satisfied |
| `sim_success` | SPICE simulation converged |
| `waveform_metrics.frequency` | Measured oscillation frequency (Hz) |
| `waveform_metrics.duty_cycle` | Measured duty cycle |
| `waveform_metrics.vpp` | Measured peak-to-peak voltage |
| `waveform_metrics.stability` | Period regularity score [0, 1] |
| `n_components` | Components placed |
| `n_invalid_actions` | Count of invalid action attempts |
| `n_convergence_fail` | Count of SPICE convergence failures |
| `reward_decomposition.*` | All 11 weighted reward sub-components |

---

## MDP Properties

| Property | Value |
|----------|-------|
| Type | Episodic, deterministic under seed |
| Observability | Partial (agent sees circuit graph + sim feedback) |
| Action space | `MultiDiscrete([4, 7, 20, 12, 12, 12, 12])` |
| Observation space | `Box(269,)` float32 |
| Reward | Shaped + sparse terminal bonus, range `[0, 1]` (server/inference) |
| Termination | FINALIZE action or `max_steps` exceeded (truncation) |
| Invalid actions | Penalised with `-0.02`, do not crash the env |
| Non-convergence | Penalised, env continues to next action |
| Reproducibility | Fully deterministic under fixed `seed` |

---

## License

MIT License. SPICE device models (2N2222, 1N4148) are public domain.
