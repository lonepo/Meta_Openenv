"""
ngspice subprocess wrapper and mock simulator for CircuitSynth.

Real mode  : writes a .cir file, calls `ngspice -b`, parses stdout.
Mock mode  : generates synthetic square waves estimated from RC time constants.
             Useful for development, CI, and environments without ngspice.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .components import (
    COMPONENT_LIBRARY,
    NODE_NAMES,
    ComponentType,
)
from .utils import detect_convergence_failure, detect_simulation_error, get_logger

logger = get_logger("circuitsynth.simulator")

# ---------------------------------------------------------------------------
# SimResult
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Result returned by a transient SPICE simulation."""
    success:         bool
    convergence:     bool
    time_array:      Optional[np.ndarray]   # seconds
    voltage_array:   Optional[np.ndarray]   # volts at output node
    error_msg:       Optional[str]
    warnings:        List[str] = field(default_factory=list)
    sim_time_seconds: float = 0.0

    @classmethod
    def failure(cls, error_msg: str, sim_time: float = 0.0) -> "SimResult":
        return cls(
            success=False, convergence=False,
            time_array=None, voltage_array=None,
            error_msg=error_msg, sim_time_seconds=sim_time,
        )

    @classmethod
    def convergence_failure(cls, sim_time: float = 0.0) -> "SimResult":
        return cls(
            success=False, convergence=False,
            time_array=None, voltage_array=None,
            error_msg="SPICE convergence failure",
            sim_time_seconds=sim_time,
        )

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "convergence": self.convergence,
            "n_time_points": len(self.time_array) if self.time_array is not None else 0,
            "error_msg": self.error_msg,
            "warnings": self.warnings,
            "sim_time_seconds": self.sim_time_seconds,
        }


# ---------------------------------------------------------------------------
# ngspice output parser
# ---------------------------------------------------------------------------

def _parse_ngspice_print(stdout: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse the tabular output produced by ngspice `.PRINT TRAN V(node)`.

    The format is:
        Index       time            v(node)
          0    0.000000e+00    5.000000e+00
          1    1.000000e-06    4.999700e+00
          ...

    Returns (time_array, voltage_array) or None if parsing fails.
    """
    times: List[float] = []
    voltages: List[float] = []

    # ngspice may output multiple tables (restart/sweep); grab all valid data rows
    in_data = False
    for line in stdout.splitlines():
        stripped = line.strip()

        # Detect header row
        if re.search(r"\btime\b", stripped, re.IGNORECASE) and re.search(r"\bv\(", stripped, re.IGNORECASE):
            in_data = True
            continue

        if not in_data:
            continue

        # Blank lines or non-numeric content end a table block
        if not stripped or stripped.startswith("#") or stripped.startswith("*"):
            in_data = False
            continue

        # Data row: "integer  float  float" (possibly with sign issues)
        parts = stripped.split()
        if len(parts) >= 3:
            try:
                times.append(float(parts[1]))
                voltages.append(float(parts[2]))
            except ValueError:
                in_data = False
                continue

    if len(times) < 10:
        return None
    return np.array(times, dtype=np.float64), np.array(voltages, dtype=np.float64)


# ---------------------------------------------------------------------------
# Mock simulator
# ---------------------------------------------------------------------------

def _estimate_rc_frequency(netlist_dict: dict) -> Tuple[float, float, float]:
    """
    Estimate oscillation frequency, supply voltage, and quality from a netlist dict.

    For a cross-coupled astable multivibrator, T ≈ 1.38 * R * C per side.
    We look for the smallest R and C that appear in the circuit.
    """
    components = netlist_dict.get("components", [])
    resistances = []
    capacitances = []
    vsource_vals = []
    n_bjt   = 0
    n_r     = 0
    n_c     = 0
    n_vsrc  = 0

    for c in components:
        ct = c.get("comp_type", "")
        v  = c.get("value", 0.0) if c.get("value") else 0.0
        if ct == "RESISTOR":
            n_r += 1
            if v > 0:
                resistances.append(v)
        elif ct == "CAPACITOR":
            n_c += 1
            if v > 0:
                capacitances.append(v)
        elif ct == "NPN_BJT":
            n_bjt += 1
        elif ct == "VSOURCE":
            n_vsrc += 1
            if v > 0:
                vsource_vals.append(v)

    # Quality heuristic: higher → more like a real oscillator
    quality = 0.0
    if n_bjt >= 2:   quality += 0.40
    if n_r >= 2:     quality += 0.20
    if n_c >= 2:     quality += 0.30
    if n_vsrc >= 1:  quality += 0.10
    quality = min(1.0, quality)

    # Frequency estimate
    if resistances and capacitances and n_bjt >= 2:
        R = float(np.median(resistances))
        C = float(np.median(capacitances))
        freq = 1.0 / (1.38 * R * C + 1e-20)
    else:
        freq = 0.0

    vcc = float(np.mean(vsource_vals)) if vsource_vals else 5.0
    return freq, vcc, quality


def _mock_simulate(
    netlist_dict: dict,
    stop_time: float,
    step_size: float,
    target_freq: float = 1000.0,
    target_vcc:  float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> SimResult:
    """
    Generate a synthetic square wave instead of calling ngspice.
    The waveform quality depends on how well-structured the circuit is.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    t0 = time.time()
    est_freq, est_vcc, quality = _estimate_rc_frequency(netlist_dict)

    t = np.arange(0, stop_time, step_size)
    n = len(t)

    if quality < 0.3 or est_freq <= 0:
        # Poor circuit → flat or noisy output
        v = np.zeros(n, dtype=np.float64) + rng.normal(0, 0.1, n)
        return SimResult(
            success=True, convergence=True,
            time_array=t, voltage_array=v,
            error_msg=None,
            warnings=["Mock mode: circuit quality too low to oscillate"],
            sim_time_seconds=time.time() - t0,
        )

    # Blend estimated frequency toward target proportional to quality
    freq = est_freq * (1 - quality) + target_freq * quality
    freq = float(np.clip(freq, 1.0, 1e6))
    vcc  = est_vcc if est_vcc > 0 else target_vcc

    # Generate square wave with slight imperfections based on quality
    duty = 0.5 + rng.uniform(-0.05, 0.05) * (1 - quality)
    phase_noise = rng.normal(0, 0.01 * (1 - quality), n)
    signal = ((t * freq + phase_noise) % 1.0) < duty
    v = signal.astype(np.float64) * vcc

    # Add noise inversely proportional to quality
    noise_std = vcc * 0.05 * (1.0 - quality)
    v += rng.normal(0, noise_std, n)

    return SimResult(
        success=True, convergence=True,
        time_array=t, voltage_array=v,
        error_msg=None,
        warnings=["Mock simulator — no real ngspice"],
        sim_time_seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Real ngspice simulator
# ---------------------------------------------------------------------------

class NgSpiceSimulator:
    """
    Wraps ngspice subprocess for transient circuit simulation.

    Parameters
    ----------
    mock : bool
        If True, use the built-in mock simulator instead of ngspice.
    ngspice_bin : str
        Path to ngspice binary (default: "ngspice" from PATH).
    timeout : float
        Subprocess timeout in seconds.
    """

    def __init__(
        self,
        mock: bool = False,
        ngspice_bin: str = "ngspice",
        timeout: float = 30.0,
    ) -> None:
        self.mock = mock
        self.ngspice_bin = ngspice_bin
        self.timeout = timeout
        self._rng: Optional[np.random.Generator] = None

        if not mock:
            if not self._check_ngspice():
                logger.warning(
                    "ngspice binary not found at '%s'. "
                    "Falling back to mock simulator. "
                    "Install ngspice or set mock=True to suppress this warning.",
                    ngspice_bin,
                )
                self.mock = True

    def set_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def _check_ngspice(self) -> bool:
        try:
            result = subprocess.run(
                [self.ngspice_bin, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ------------------------------------------------------------------

    def run_transient(
        self,
        netlist_str: str,
        stop_time: float,
        step_size: float,
        netlist_dict: Optional[dict] = None,
        target_freq: float = 1000.0,
    ) -> SimResult:
        """
        Run a transient simulation.

        Parameters
        ----------
        netlist_str  : SPICE netlist string
        stop_time    : total simulation time (seconds)
        step_size    : time step (seconds)
        netlist_dict : raw netlist dict for mock mode quality estimation
        target_freq  : expected frequency for mock mode calibration

        Returns
        -------
        SimResult
        """
        if self.mock:
            return _mock_simulate(
                netlist_dict=netlist_dict or {},
                stop_time=stop_time,
                step_size=step_size,
                target_freq=target_freq,
                rng=self._rng,
            )
        return self._run_ngspice(netlist_str)

    # ------------------------------------------------------------------

    def _run_ngspice(self, netlist_str: str) -> SimResult:
        t0 = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            cir_path = os.path.join(tmpdir, "circuit.cir")
            with open(cir_path, "w") as f:
                f.write(netlist_str)

            try:
                proc = subprocess.run(
                    [self.ngspice_bin, "-b", cir_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return SimResult.failure(
                    f"ngspice timed out after {self.timeout}s",
                    sim_time=time.time() - t0,
                )
            except FileNotFoundError:
                return SimResult.failure(
                    f"ngspice binary not found: {self.ngspice_bin}",
                    sim_time=time.time() - t0,
                )

        elapsed = time.time() - t0
        stdout = proc.stdout
        stderr = proc.stderr

        # Check for convergence failure
        combined = stdout + stderr
        if detect_convergence_failure(combined):
            return SimResult.convergence_failure(sim_time=elapsed)
        if proc.returncode != 0 and detect_simulation_error(combined):
            return SimResult.failure(
                f"ngspice returned code {proc.returncode}: {stderr[:400]}",
                sim_time=elapsed,
            )

        # Parse waveform data
        parsed = _parse_ngspice_print(stdout)
        if parsed is None:
            return SimResult.failure(
                "Could not parse ngspice output — check .PRINT statement",
                sim_time=elapsed,
            )

        time_arr, volt_arr = parsed
        warnings = []
        if "warning" in combined.lower():
            # Extract first warning line
            for line in combined.splitlines():
                if "warning" in line.lower():
                    warnings.append(line.strip())
                    break

        return SimResult(
            success=True,
            convergence=True,
            time_array=time_arr,
            voltage_array=volt_arr,
            error_msg=None,
            warnings=warnings,
            sim_time_seconds=elapsed,
        )
