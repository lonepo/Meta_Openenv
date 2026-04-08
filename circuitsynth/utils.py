"""Utility helpers: seeding, SI formatting, logging, convergence detection."""

import sys
import random
import logging
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "circuitsynth") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s — %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python random, numpy, and return a numpy Generator for the env."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# SI value formatting
# ---------------------------------------------------------------------------

_SI_PREFIXES = [
    (1e12,  "T"),
    (1e9,   "G"),
    (1e6,   "M"),
    (1e3,   "k"),
    (1.0,   ""),
    (1e-3,  "m"),
    (1e-6,  "µ"),
    (1e-9,  "n"),
    (1e-12, "p"),
    (1e-15, "f"),
]


def format_si(value: float, unit: str = "", precision: int = 3) -> str:
    """Return a human-readable SI-prefixed string, e.g. 47000 Ω → '47.000 kΩ'."""
    if value == 0.0:
        return f"0 {unit}"
    abs_val = abs(value)
    for magnitude, prefix in _SI_PREFIXES:
        if abs_val >= magnitude:
            formatted = f"{value / magnitude:.{precision}g}"
            return f"{formatted} {prefix}{unit}"
    return f"{value:.{precision}e} {unit}"


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------

CONVERGENCE_FAILURE_TOKENS = [
    "convergence",
    "singular",
    "iteration limit",
    "fatal",
    "error:",
    "timestep too small",
    "gmin stepping",
    "source stepping",
]


def detect_convergence_failure(stderr_text: str) -> bool:
    """Return True if ngspice stderr signals a convergence failure."""
    lower = stderr_text.lower()
    return any(tok in lower for tok in CONVERGENCE_FAILURE_TOKENS)


def detect_simulation_error(stderr_text: str) -> bool:
    """Return True if ngspice stderr signals a fatal error (not convergence)."""
    lower = stderr_text.lower()
    return "error" in lower or "fatal" in lower


# ---------------------------------------------------------------------------
# Waveform helpers
# ---------------------------------------------------------------------------

def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-20:
        return default
    return num / den


def normalize_to_unit(value: float, lo: float, hi: float) -> float:
    """Linearly map value from [lo, hi] → [0, 1], clamped."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def exponential_score(error_ratio: float, scale: float = 5.0) -> float:
    """Score in [0,1] that decays exponentially with relative error."""
    return float(np.exp(-scale * max(0.0, error_ratio)))
