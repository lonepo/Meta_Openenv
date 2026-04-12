#!/usr/bin/env python3
"""
inference.py — OpenEnv-compliant LLM-based inference script for CircuitSynth-SquareWave.

The agent (LLM via OpenAI client) reads a text observation describing the
current circuit state and returns a JSON action to build or finalize the circuit.

Structured stdout format:
  [START] {"task": ..., "env": ..., "model": ...}
  [STEP]  {"step": ..., "action": ..., "reward": ..., "done": ..., "error": ...}
  [END]   {"success": ..., "steps": ..., "score": ..., "rewards": [...]}

Environment variables required:
  API_BASE_URL  — OpenAI-compatible API base URL
  MODEL_NAME    — Model identifier
  HF_TOKEN      — Hugging Face / API key

Usage:
  python inference.py
  TASK_NAME=squarewave-hard python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import re
import asyncio
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
API_KEY      = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "hf_placeholder")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.environ.get("TASK_NAME",    "squarewave-easy")
BENCHMARK    = "CircuitSynth-SquareWave"

MOCK_SIM = os.environ.get("MOCK_SIM", "true").lower() in ("1", "true", "yes")

# Episode config
MAX_STEPS         = 20    # must complete within 20 min total; kept short per step
MAX_TOTAL_REWARD  = 1.0   # a single perfect FINALIZE gives 1.0
SUCCESS_SCORE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Structured logging — strict [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    payload = {"task": task, "env": env, "model": model}
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    payload = {
        "step":   step,
        "action": action,
        "reward": round(float(reward), 6),
        "done":   bool(done),
        "error":  error,
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    payload = {
        "success": bool(success),
        "steps":   steps,
        "score":   round(float(score), 6),
        "rewards": [round(float(r), 6) for r in rewards],
    }
    print(f"[END] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# In-process CircuitSynth environment wrapper
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuitsynth.env import CircuitSynthEnv
from circuitsynth.components import NODE_NAMES, COMPONENT_LIBRARY, ComponentType
from circuitsynth.action_space import ActionType


def _format_observation(env: CircuitSynthEnv) -> str:
    """
    Convert the current environment state into a human-readable text
    that the LLM can parse as context for its next action.
    """
    task    = env.task
    netlist = env._netlist
    metrics = env._last_metrics
    tgt     = task.target

    lines = [
        f"=== CircuitSynth Task: {env.task_id} ===",
        f"TARGET  : {tgt.frequency:.0f} Hz square wave, "
        f"{tgt.duty_cycle*100:.0f}% duty cycle, {tgt.amplitude:.1f} V amplitude",
        f"TOLERANCES: freq ±{tgt.freq_tol*100:.0f}%  "
        f"DC ±{tgt.dc_tol*100:.0f}%  amp ±{tgt.amp_tol*100:.0f}%",
        "",
        f"CIRCUIT ({len(netlist)}/{task.max_components} components):",
    ]

    if not netlist.components:
        lines.append("  (empty — no components placed yet)")
    else:
        for pc in netlist.components:
            conn = {t: NODE_NAMES[n] for t, n in pc.connections.items()}
            from circuitsynth.utils import format_si
            from circuitsynth.components import COMPONENT_LIBRARY
            spec = COMPONENT_LIBRARY[pc.comp_type]
            val_str = format_si(pc.value, spec.unit) if pc.value else "—"
            conn_str = "  ".join(f"{t}={v}" for t, v in conn.items())
            lines.append(f"  {pc.comp_id:6s} {pc.comp_type.name:12s} {val_str:12s}  {conn_str}")

    lines.append("")
    if metrics.sim_success and metrics.frequency > 0:
        lines.append("LAST SIM:")
        lines.append(f"  Frequency : {metrics.frequency:.1f} Hz  (target {tgt.frequency:.0f} Hz)")
        lines.append(f"  Duty cycle: {metrics.duty_cycle:.3f}   (target {tgt.duty_cycle:.2f})")
        lines.append(f"  Vpp       : {metrics.vpp:.2f} V     (target {tgt.amplitude:.1f} V)")
        lines.append(f"  Stability : {metrics.stability:.3f}")
    else:
        lines.append("LAST SIM: Not yet run (call FINALIZE to simulate)")

    lines.append("")
    lines.append("AVAILABLE ACTIONS (respond with exactly one JSON object):")
    lines.append('  ADD component:')
    lines.append('    {"action":"ADD","component":"RESISTOR","node_a":"VCC","node_b":"N1","value_idx":13}')
    lines.append('    {"action":"ADD","component":"CAPACITOR","node_a":"N1","node_b":"N2","value_idx":12}')
    lines.append('    {"action":"ADD","component":"NPN_BJT","node_a":"N1","node_b":"N3","node_c":"GND","value_idx":0}')
    lines.append('    {"action":"ADD","component":"VSOURCE","node_a":"VCC","node_b":"GND","value_idx":5}')
    lines.append('  REMOVE by index:')
    lines.append('    {"action":"REMOVE","index":0}')
    lines.append('  FINALIZE (run simulation):')
    lines.append('    {"action":"FINALIZE"}')
    lines.append("")
    lines.append(f"NODES   : VCC  GND  N1  N2  N3  N4  N5  N6  N7  N8  N9  N10")
    lines.append(f"RESISTOR values (value_idx 0-19): 10Ω … 10MΩ (log-spaced)")
    lines.append(f"CAPACITOR values (value_idx 0-19): 1pF … 1mF (log-spaced)")
    lines.append(f"VSOURCE values (value_idx 0-19): 1V … 15V")
    lines.append(f"BUDGET  : {task.max_components - len(netlist)} components remaining, "
                 f"{task.max_steps - env._step_count} steps remaining")
    lines.append("")
    lines.append("HINT: A classic 2-BJT astable multivibrator needs:")
    lines.append("  1x VSOURCE(VCC→GND), 2x RESISTOR(VCC→N1,N2) collector,")
    lines.append("  2x RESISTOR(VCC→N3,N4) base, 2x CAPACITOR cross-coupled,")
    lines.append("  2x NPN_BJT(C=N1|N2, B=N3|N4, E=GND)")
    lines.append("  Then FINALIZE to simulate.")

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a circuit design agent. Your task is to build a transistor-based square-wave \
oscillator circuit step-by-step. Study the current circuit state and output EXACTLY ONE \
JSON object (no extra text, no markdown, no code fences) describing your next action.

Rules:
- Use only nodes: VCC, GND, N1..N10
- Do NOT connect a component's two terminals to the same node
- A BJT must have 3 distinct nodes: node_a=Collector, node_b=Base, node_c=Emitter
- Call FINALIZE when you believe the circuit is complete enough to simulate
- Prefer the classic astable topology: 2 BJTs, 2 collector R, 2 base R, 2 cross-coupling C, 1 VCC supply
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    step: int,
    observation: str,
    last_reward: float,
    history: List[str],
) -> str:
    history_str = "\n".join(history[-6:]) if history else "None"
    user_msg = (
        f"Step {step} | Last reward: {last_reward:+.4f}\n\n"
        f"Recent history:\n{history_str}\n\n"
        f"Current state:\n{observation}\n\n"
        f"Output your next action as a single JSON object:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.2,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"action":"NO_OP"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action":"FINALIZE"}'   # safe fallback


# ---------------------------------------------------------------------------
# Action parser: LLM text → environment action vector
# ---------------------------------------------------------------------------

def _node_idx(name: str) -> int:
    name = str(name).strip().upper()
    return NODE_NAMES.index(name) if name in NODE_NAMES else 1   # default GND


COMP_NAME_MAP = {
    "RESISTOR":  ComponentType.RESISTOR,
    "R":         ComponentType.RESISTOR,
    "CAPACITOR": ComponentType.CAPACITOR,
    "C":         ComponentType.CAPACITOR,
    "CAP":       ComponentType.CAPACITOR,
    "NPN_BJT":   ComponentType.NPN_BJT,
    "BJT":       ComponentType.NPN_BJT,
    "NPN":       ComponentType.NPN_BJT,
    "DIODE":     ComponentType.DIODE,
    "D":         ComponentType.DIODE,
    "VSOURCE":   ComponentType.VSOURCE,
    "VCC_SRC":   ComponentType.VSOURCE,
    "V":         ComponentType.VSOURCE,
    "SWITCH":    ComponentType.SWITCH,
    "S":         ComponentType.SWITCH,
}


import numpy as np

def parse_llm_action(text: str) -> np.ndarray:
    """
    Parse LLM text output into a 7-element MultiDiscrete action vector.
    Tries JSON first; falls back to keyword extraction.
    Returns NO_OP on parse failure.
    """
    # Extract JSON from the text (handles markdown code fences, trailing text)
    json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not json_match:
        return np.array([3, 0, 0, 0, 1, 1, 0], dtype=np.int64)   # NO_OP

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return np.array([3, 0, 0, 0, 1, 1, 0], dtype=np.int64)

    act = obj.get("action", "NO_OP").upper()

    if act == "FINALIZE":
        return np.array([2, 0, 0, 0, 1, 1, 0], dtype=np.int64)

    if act == "REMOVE":
        idx = int(obj.get("index", 0))
        return np.array([1, 0, 0, 0, 1, 1, idx], dtype=np.int64)

    if act in ("ADD", "ADD_COMPONENT"):
        comp_str = str(obj.get("component", obj.get("type", "RESISTOR"))).upper()
        comp_type = COMP_NAME_MAP.get(comp_str, ComponentType.RESISTOR)
        val_idx   = int(obj.get("value_idx", obj.get("val_idx", 10)))
        val_idx   = max(0, min(val_idx, 19))
        node_a    = _node_idx(obj.get("node_a", "VCC"))
        node_b    = _node_idx(obj.get("node_b", "GND"))
        node_c    = _node_idx(obj.get("node_c", "GND"))
        return np.array([0, int(comp_type), val_idx, node_a, node_b, node_c, 0],
                        dtype=np.int64)

    # NO_OP fallback
    return np.array([3, 0, 0, 0, 1, 1, 0], dtype=np.int64)


# ---------------------------------------------------------------------------
# Main episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    client: OpenAI,
    task_id: str,
    seed: int = 42,
) -> dict:
    env = CircuitSynthEnv(task_id=task_id, seed=seed, mock_sim=MOCK_SIM)
    obs, info = env.reset(seed=seed)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken  = 0
    score        = 0.0
    success      = False
    last_reward  = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            done = env._terminated or env._truncated
            if done:
                break

            # Get text observation
            observation = _format_observation(env)

            # LLM decides next action
            llm_text   = get_model_action(client, step, observation, last_reward, history)
            action_vec = parse_llm_action(llm_text)

            # Step environment
            try:
                obs, raw_reward, terminated, truncated, info = env.step(action_vec)
                done  = terminated or truncated
                error = None
            except Exception as exc:
                raw_reward = 0.0
                done       = False
                error      = str(exc)
                print(f"[DEBUG] env.step error: {exc}", flush=True)

            # Normalize reward to [0, 1]
            reward     = float(np.clip((raw_reward + 1.0) / 2.0, 0.0, 1.0))
            last_reward = reward

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=llm_text, reward=reward, done=done, error=error)
            history.append(
                f"Step {step}: {llm_text[:80]!r} → reward {reward:+.4f}"
            )

            if done:
                break

        # Final score: max single-step reward (FINALIZE dominates)
        score   = max(rewards) if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"score": score, "success": success, "steps": steps_taken}


def main():
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        asyncio.run(run_episode(client=client, task_id=TASK_NAME, seed=42))
    except Exception as exc:
        print(f"[ERROR] inference.py failed: {exc}", file=sys.stderr, flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(1)


if __name__ == "__main__":
    main()
