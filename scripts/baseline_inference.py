#!/usr/bin/env python3
"""
baseline_inference.py — Reproducible random baseline rollout for CircuitSynth.

Runs a random policy for N episodes and logs:
  - Episode return
  - Success rate (all_thresholds_met)
  - Waveform error metrics (frequency, duty cycle, amplitude errors)
  - Simulation convergence rate
  - Average component count
  - Invalid-action frequency
  - Number of non-convergent runs

Output: JSON file + console summary.

Usage:
    python scripts/baseline_inference.py
    python scripts/baseline_inference.py --task squarewave-medium --episodes 50 --seed 0
    python scripts/baseline_inference.py --mock                    # no ngspice needed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure package is importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from circuitsynth import CircuitSynthEnv, TASK_REGISTRY
from circuitsynth.waveform import WaveformMetrics


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------

def random_policy(env: CircuitSynthEnv, rng: np.random.Generator) -> np.ndarray:
    """Sample a random valid action from the environment's action space."""
    action = env.action_space.sample()
    # With 30% probability, choose FINALIZE to keep episodes short
    if rng.random() < 0.05:
        action[0] = 2   # FINALIZE
    return action


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_episode(env: CircuitSynthEnv, seed: int, rng: np.random.Generator) -> dict:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = random_policy(env, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

    m   = env._last_metrics
    d   = env._last_reward_decomp
    sim = env._last_sim

    freq_err = (
        abs(m.frequency - env.task.target.frequency) / max(env.task.target.frequency, 1e-9)
        if m.sim_success else float("nan")
    )
    dc_err = (
        abs(m.duty_cycle - env.task.target.duty_cycle)
        if m.sim_success else float("nan")
    )
    amp_err = (
        abs(m.vpp - env.task.target.amplitude) / max(env.task.target.amplitude, 1e-9)
        if m.sim_success else float("nan")
    )

    return {
        "total_reward":        total_reward,
        "n_steps":             step,
        "n_components":        len(env._netlist),
        "n_invalid_actions":   env._n_invalid_actions,
        "n_convergence_fail":  env._n_convergence_fail,
        "sim_success":         sim.success if sim else False,
        "all_thresholds_met":  d.all_thresholds_met,
        "freq_error":          freq_err,
        "dc_error":            dc_err,
        "amp_error":           amp_err,
        "measured_frequency":  m.frequency,
        "measured_duty_cycle": m.duty_cycle,
        "measured_vpp":        m.vpp,
        "stability":           m.stability,
        "reward_decomposition": d.to_dict(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CircuitSynth — random baseline inference script"
    )
    parser.add_argument("--task",     default="squarewave-easy",
                        choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--mock",     action="store_true",
                        help="Use mock simulator (no ngspice required)")
    parser.add_argument("--output",   default="baseline_results.json",
                        help="Path to write JSON results")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CircuitSynth Baseline Inference")
    print(f"  Task     : {args.task}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Seed     : {args.seed}")
    print(f"  Simulator: {'MOCK' if args.mock else 'ngspice'}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(args.seed)
    env = CircuitSynthEnv(
        task_id=args.task,
        seed=args.seed,
        mock_sim=args.mock,
    )

    results = []
    t_start = time.time()

    for ep in range(args.episodes):
        ep_seed = int(rng.integers(0, 2**31))
        record  = run_episode(env, seed=ep_seed, rng=rng)
        results.append(record)

        if args.verbose or (ep + 1) % max(1, args.episodes // 10) == 0:
            print(
                f"  ep {ep+1:4d}/{args.episodes}  "
                f"reward={record['total_reward']:+.4f}  "
                f"freq_err={record['freq_error']:.3f}  "
                f"sim={'OK' if record['sim_success'] else 'FAIL'}  "
                f"thresh={'✓' if record['all_thresholds_met'] else '✗'}"
            )

    elapsed = time.time() - t_start

    # Aggregate stats
    rewards     = np.array([r["total_reward"]       for r in results])
    freq_errs   = np.array([r["freq_error"]         for r in results if np.isfinite(r["freq_error"])])
    dc_errs     = np.array([r["dc_error"]            for r in results if np.isfinite(r["dc_error"])])
    amp_errs    = np.array([r["amp_error"]           for r in results if np.isfinite(r["amp_error"])])
    n_comps     = np.array([r["n_components"]        for r in results])
    inv_acts    = np.array([r["n_invalid_actions"]   for r in results])
    conv_fails  = np.array([r["n_convergence_fail"]  for r in results])
    sim_oks     = np.array([r["sim_success"]         for r in results])
    thresholds  = np.array([r["all_thresholds_met"]  for r in results])

    summary = {
        "task_id":              args.task,
        "n_episodes":           args.episodes,
        "seed":                 args.seed,
        "mock_sim":             args.mock,
        "elapsed_seconds":      round(elapsed, 2),
        "mean_return":          float(rewards.mean()),
        "std_return":           float(rewards.std()),
        "max_return":           float(rewards.max()),
        "min_return":           float(rewards.min()),
        "success_rate":         float(thresholds.mean()),
        "sim_convergence_rate": float(sim_oks.mean()),
        "mean_freq_error":      float(freq_errs.mean())  if len(freq_errs)  else None,
        "mean_dc_error":        float(dc_errs.mean())    if len(dc_errs)    else None,
        "mean_amp_error":       float(amp_errs.mean())   if len(amp_errs)   else None,
        "mean_n_components":    float(n_comps.mean()),
        "mean_invalid_acts":    float(inv_acts.mean()),
        "mean_convergence_fail":float(conv_fails.mean()),
        "per_episode":          results,
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean return         : {summary['mean_return']:+.4f} ± {summary['std_return']:.4f}")
    print(f"  Success rate        : {summary['success_rate']:.1%}")
    print(f"  Sim convergence     : {summary['sim_convergence_rate']:.1%}")
    print(f"  Mean freq error     : {summary['mean_freq_error']}")
    print(f"  Mean duty-cycle err : {summary['mean_dc_error']}")
    print(f"  Mean amplitude err  : {summary['mean_amp_error']}")
    print(f"  Avg components      : {summary['mean_n_components']:.1f}")
    print(f"  Avg invalid actions : {summary['mean_invalid_acts']:.1f}")
    print(f"  Total elapsed       : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    out_path = Path(args.output).resolve()
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
