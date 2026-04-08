#!/usr/bin/env python3
"""
evaluate.py — Offline evaluation for trained CircuitSynth policies.

Supports:
  - Loading a Stable-Baselines3 model and evaluating it
  - Loading a JSON policy script (list of action dicts per episode)
  - Running a known-good reference circuit to validate the reward function

Usage:
    # SB3 model evaluation
    python scripts/evaluate.py --model path/to/model.zip --task squarewave-hard

    # Reference circuit validation (sanity check)
    python scripts/evaluate.py --reference --task squarewave-easy --mock

    # Scripted action sequence from JSON
    python scripts/evaluate.py --script path/to/actions.json --task squarewave-medium
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from circuitsynth import CircuitSynthEnv, TASK_REGISTRY
from circuitsynth.components import ComponentType, NODE_INDEX


# ---------------------------------------------------------------------------
# Reference circuit for task validation
# ---------------------------------------------------------------------------

# Classic symmetric astable multivibrator:
# 2× NPN BJTs, 2× collector resistors (1 kΩ), 2× base resistors (47 kΩ),
# 2× cross-coupling capacitors (15 nF), 1× VCC supply (5 V)

REFERENCE_CIRCUITS = {
    "squarewave-easy": [
        # VCC supply: VCC (+) to GND (-)
        {"action_type": "ADD_COMPONENT", "component_type": "VSOURCE",
         "value_idx": 5, "node_a": "VCC", "node_b": "GND", "node_c": "GND"},
        # Collector resistor R1: VCC → N1 (1 kΩ ≈ idx 5 in log scale)
        {"action_type": "ADD_COMPONENT", "component_type": "RESISTOR",
         "value_idx": 5, "node_a": "VCC", "node_b": "N1", "node_c": "GND"},
        # Collector resistor R2: VCC → N2
        {"action_type": "ADD_COMPONENT", "component_type": "RESISTOR",
         "value_idx": 5, "node_a": "VCC", "node_b": "N2", "node_c": "GND"},
        # Base resistor R3: VCC → N3 (47 kΩ ≈ idx 13)
        {"action_type": "ADD_COMPONENT", "component_type": "RESISTOR",
         "value_idx": 13, "node_a": "VCC", "node_b": "N3", "node_c": "GND"},
        # Base resistor R4: VCC → N4
        {"action_type": "ADD_COMPONENT", "component_type": "RESISTOR",
         "value_idx": 13, "node_a": "VCC", "node_b": "N4", "node_c": "GND"},
        # Cross-coupling capacitor C1: N1 → N4 (15 nF ≈ idx 12)
        {"action_type": "ADD_COMPONENT", "component_type": "CAPACITOR",
         "value_idx": 12, "node_a": "N1", "node_b": "N4", "node_c": "GND"},
        # Cross-coupling capacitor C2: N2 → N3
        {"action_type": "ADD_COMPONENT", "component_type": "CAPACITOR",
         "value_idx": 12, "node_a": "N2", "node_b": "N3", "node_c": "GND"},
        # BJT Q1: collector=N1, base=N3, emitter=GND
        {"action_type": "ADD_COMPONENT", "component_type": "NPN_BJT",
         "value_idx": 0, "node_a": "N1", "node_b": "N3", "node_c": "GND"},
        # BJT Q2: collector=N2, base=N4, emitter=GND
        {"action_type": "ADD_COMPONENT", "component_type": "NPN_BJT",
         "value_idx": 0, "node_a": "N2", "node_b": "N4", "node_c": "GND"},
        # FINALIZE
        {"action_type": "FINALIZE"},
    ],
}
# Reuse easy reference for other tasks (different targets, same topology)
REFERENCE_CIRCUITS["squarewave-medium"] = REFERENCE_CIRCUITS["squarewave-easy"]
REFERENCE_CIRCUITS["squarewave-hard"]   = REFERENCE_CIRCUITS["squarewave-easy"]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_scripted_episode(env: CircuitSynthEnv, actions: list) -> dict:
    """Execute a fixed sequence of action dicts and return result."""
    obs, info = env.reset()
    total_reward = 0.0
    step_rewards = []

    for act_dict in actions:
        if act_dict.get("action_type") == "FINALIZE":
            obs, r, terminated, truncated, info = env.step_dict(
                {"action_type": "FINALIZE"}
            )
        else:
            obs, r, terminated, truncated, info = env.step_dict(act_dict)

        total_reward += r
        step_rewards.append(r)
        if terminated or truncated:
            break

    return {
        "total_reward":        total_reward,
        "step_rewards":        step_rewards,
        "n_components":        len(env._netlist),
        "sim_success":         env._last_sim.success if env._last_sim else False,
        "all_thresholds_met":  env._last_reward_decomp.all_thresholds_met,
        "waveform_metrics":    env._last_metrics.to_dict(),
        "reward_decomposition": env._last_reward_decomp.to_dict(),
        "netlist":             env._netlist.to_dict(),
    }


def evaluate_sb3_model(model_path: str, task_id: str, n_episodes: int,
                        mock: bool, seed: int) -> dict:
    """Evaluate a Stable-Baselines3 policy."""
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError:
        print("stable-baselines3 not installed. Install with: pip install stable-baselines3")
        sys.exit(1)

    env   = CircuitSynthEnv(task_id=task_id, seed=seed, mock_sim=mock)
    model = PPO.load(model_path)

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        ep_reward  = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += r
        rewards.append(ep_reward)

    return {
        "model":       model_path,
        "task_id":     task_id,
        "n_episodes":  n_episodes,
        "mean_return": float(np.mean(rewards)),
        "std_return":  float(np.std(rewards)),
        "max_return":  float(np.max(rewards)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CircuitSynth offline evaluation")
    parser.add_argument("--task",       default="squarewave-easy",
                        choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--episodes",   type=int, default=1)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--mock",       action="store_true")
    parser.add_argument("--reference",  action="store_true",
                        help="Run reference circuit to validate reward function")
    parser.add_argument("--script",     default=None,
                        help="Path to JSON file with list of action dicts")
    parser.add_argument("--model",      default=None,
                        help="Path to SB3 model .zip file")
    parser.add_argument("--output",     default="eval_results.json")
    args = parser.parse_args()

    env = CircuitSynthEnv(task_id=args.task, seed=args.seed, mock_sim=args.mock)

    if args.reference:
        circuit = REFERENCE_CIRCUITS.get(args.task)
        if circuit is None:
            print(f"No reference circuit for task {args.task}")
            sys.exit(1)

        print(f"\nRunning reference circuit validation for task: {args.task}\n")
        result = run_scripted_episode(env, circuit)

        print(f"Total reward     : {result['total_reward']:+.4f}")
        print(f"Sim success      : {result['sim_success']}")
        print(f"Thresholds met   : {result['all_thresholds_met']}")
        m = result["waveform_metrics"]
        t = env.task.target
        print(f"Frequency        : {m['frequency']:.1f} Hz  (target={t.frequency} Hz)")
        print(f"Duty cycle       : {m['duty_cycle']:.3f}   (target={t.duty_cycle})")
        print(f"Vpp              : {m['vpp']:.2f} V   (target={t.amplitude} V)")
        print(f"Stability        : {m['stability']:.3f}")
        print("\nReward decomposition:")
        for k, v in result["reward_decomposition"].items():
            if isinstance(v, float):
                print(f"  {k:35s}: {v:+.4f}")

        out = Path(args.output)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {out}")

    elif args.script:
        with open(args.script) as f:
            actions = json.load(f)
        result = run_scripted_episode(env, actions)
        print(json.dumps(result, indent=2, default=str))
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)

    elif args.model:
        result = evaluate_sb3_model(
            args.model, args.task, args.episodes, args.mock, args.seed
        )
        print(json.dumps(result, indent=2))
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
