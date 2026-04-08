#!/usr/bin/env python3
"""
server/app.py — Self-contained FastAPI server for CircuitSynth-SquareWave.

This is the canonical entry point for:
  - Multi-mode deployment validator  (server.app:app, server.app:main)
  - Dockerfile uvicorn               (server.app:app)
  - Direct execution                 (python server/app.py)

Endpoints:
  GET  /health                → {"status": "ok"}
  GET  /                      → same as /health
  GET  /reset                 → ResetResponse  (no-body form)
  POST /reset                 → ResetResponse
  POST /step                  → StepResponse
  GET  /state/{session_id}    → StateResponse
  GET  /tasks                 → list of task specs
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Dict, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is importable regardless of CWD
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from circuitsynth.env import CircuitSynthEnv
from circuitsynth.tasks import TASK_REGISTRY, list_tasks
from circuitsynth.components import NODE_NAMES

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CircuitSynth-SquareWave",
    description=(
        "OpenEnv-compliant RL environment for electronic circuit synthesis. "
        "Agent learns to design switching circuits that produce target square waveforms."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_sessions: Dict[str, CircuitSynthEnv] = {}
_MAX_SESSIONS = 32

MOCK_SIM = os.environ.get("MOCK_SIM", "true").lower() in ("1", "true", "yes")


def _get_or_create_env(session_id: str, task_id: str, seed: int) -> CircuitSynthEnv:
    if session_id not in _sessions:
        if len(_sessions) >= _MAX_SESSIONS:
            oldest = next(iter(_sessions))
            del _sessions[oldest]
        _sessions[session_id] = CircuitSynthEnv(
            task_id=task_id, seed=seed, mock_sim=MOCK_SIM
        )
    return _sessions[session_id]


def _format_obs_text(env: CircuitSynthEnv) -> str:
    task    = env.task
    netlist = env._netlist
    metrics = env._last_metrics
    tgt     = task.target

    parts = [
        f"Task: {env.task_id} | Target: {tgt.frequency:.0f} Hz, "
        f"{tgt.duty_cycle*100:.0f}% DC, {tgt.amplitude:.1f}V | "
        f"Budget: {len(netlist)}/{task.max_components} components, "
        f"step {env._step_count}/{task.max_steps}",
    ]

    if netlist.components:
        comp_lines = []
        for pc in netlist.components:
            conn = {t: NODE_NAMES[n] for t, n in pc.connections.items()}
            conn_str = " ".join(f"{t}={v}" for t, v in conn.items())
            comp_lines.append(f"{pc.comp_id}({pc.comp_type.name},{pc.value:.3g}) {conn_str}")
        parts.append("Circuit: " + "; ".join(comp_lines))
    else:
        parts.append("Circuit: empty")

    if metrics.sim_success and metrics.frequency > 0:
        parts.append(
            f"LastSim: freq={metrics.frequency:.1f}Hz dc={metrics.duty_cycle:.3f} "
            f"Vpp={metrics.vpp:.2f}V stability={metrics.stability:.3f}"
        )

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id:    str          = Field("squarewave-easy", description="Task identifier")
    seed:       int          = Field(42,   description="RNG seed")
    session_id: Optional[str]= Field(None, description="Session ID (auto-generated if None)")


class StepRequest(BaseModel):
    session_id: str            = Field(..., description="Session ID from /reset")
    action:     Dict[str, Any] = Field(
        ...,
        description=(
            'e.g. {"action":"ADD","component":"NPN_BJT","node_a":"N1",'
            '"node_b":"N3","node_c":"GND","value_idx":0} | '
            '{"action":"FINALIZE"} | {"action":"REMOVE","index":0}'
        )
    )


class ObservationModel(BaseModel):
    echoed_message:   str
    circuit_json:     dict
    waveform_metrics: dict
    budget_remaining: int
    steps_remaining:  int
    has_finalized:    bool


class ResetResponse(BaseModel):
    session_id:  str
    task_id:     str
    observation: ObservationModel
    done:        bool  = False
    reward:      float = 0.0


class StepResponse(BaseModel):
    session_id:           str
    observation:          ObservationModel
    reward:               float
    done:                 bool
    truncated:            bool
    invalid_action:       bool
    reward_decomposition: dict
    info:                 dict


class StateResponse(BaseModel):
    session_id: str
    state:      dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(env: CircuitSynthEnv) -> ObservationModel:
    return ObservationModel(
        echoed_message=_format_obs_text(env),
        circuit_json=env._netlist.to_dict(),
        waveform_metrics=env._last_metrics.to_dict(),
        budget_remaining=env.task.max_components - len(env._netlist),
        steps_remaining=env.task.max_steps - env._step_count,
        has_finalized=env._has_finalized,
    )


def _parse_action_dict(obj: dict) -> np.ndarray:
    from circuitsynth.components import ComponentType

    COMP_MAP = {
        "RESISTOR": ComponentType.RESISTOR,  "R":   ComponentType.RESISTOR,
        "CAPACITOR":ComponentType.CAPACITOR, "C":   ComponentType.CAPACITOR,
        "NPN_BJT":  ComponentType.NPN_BJT,   "BJT": ComponentType.NPN_BJT,
        "NPN":      ComponentType.NPN_BJT,
        "DIODE":    ComponentType.DIODE,     "D":   ComponentType.DIODE,
        "VSOURCE":  ComponentType.VSOURCE,   "V":   ComponentType.VSOURCE,
        "SWITCH":   ComponentType.SWITCH,
    }

    def node(name, default=1):
        n = str(name).strip().upper()
        return NODE_NAMES.index(n) if n in NODE_NAMES else default

    act = str(obj.get("action", "NO_OP")).upper()

    if act == "FINALIZE":
        return np.array([2, 0, 0, 0, 1, 1, 0], dtype=np.int64)
    if act == "NO_OP":
        return np.array([3, 0, 0, 0, 1, 1, 0], dtype=np.int64)
    if act in ("REMOVE", "REMOVE_COMPONENT"):
        idx = int(obj.get("index", obj.get("remove_idx", 0)))
        return np.array([1, 0, 0, 0, 1, 1, idx], dtype=np.int64)
    if act in ("ADD", "ADD_COMPONENT"):
        comp_str  = str(obj.get("component", obj.get("type", "RESISTOR"))).upper()
        comp_type = COMP_MAP.get(comp_str, ComponentType.RESISTOR)
        val_idx   = max(0, min(int(obj.get("value_idx", 10)), 19))
        na = node(obj.get("node_a", "VCC"), 0)
        nb = node(obj.get("node_b", "GND"), 1)
        nc = node(obj.get("node_c", "GND"), 1)
        return np.array([0, int(comp_type), val_idx, na, nb, nc, 0], dtype=np.int64)

    return np.array([3, 0, 0, 0, 1, 1, 0], dtype=np.int64)


def _normalize_reward(raw: float) -> float:
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "env":      "CircuitSynth-SquareWave",
        "version":  "1.0.0",
        "tasks":    list(TASK_REGISTRY.keys()),
        "mock_sim": MOCK_SIM,
    }


@app.get("/")
async def root():
    return await health()


@app.get("/reset",  response_model=ResetResponse)
@app.post("/reset", response_model=ResetResponse)
async def reset(req: Optional[ResetRequest] = Body(default=None)):
    """Reset an episode. Body is optional — all fields have defaults."""
    if req is None:
        req = ResetRequest()
    session_id = req.session_id or str(uuid.uuid4())
    env = _get_or_create_env(session_id, req.task_id, req.seed)

    try:
        _, info = env.reset(seed=req.seed, task_id=req.task_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ResetResponse(
        session_id=session_id,
        task_id=req.task_id,
        observation=_make_obs(env),
        done=False,
        reward=0.0,
    )


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call /reset first."
        )
    if env._terminated or env._truncated:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode."
        )

    action_vec = _parse_action_dict(req.action)

    try:
        _, raw_reward, terminated, truncated, info = env.step(action_vec)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    reward = _normalize_reward(raw_reward)
    done   = terminated or truncated

    return StepResponse(
        session_id=req.session_id,
        observation=_make_obs(env),
        reward=reward,
        done=done,
        truncated=truncated,
        invalid_action=info.get("invalid_action", False),
        reward_decomposition=info.get("reward_decomposition", {}),
        info={k: v for k, v in info.items() if k != "reward_decomposition"},
    )


@app.get("/state/{session_id}", response_model=StateResponse)
async def get_state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return StateResponse(session_id=session_id, state=env.state())


@app.get("/tasks")
async def get_tasks():
    return {"tasks": list_tasks()}


# ---------------------------------------------------------------------------
# Entry point — required by deployment validator
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the uvicorn server. Called by 'server' console script."""
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
