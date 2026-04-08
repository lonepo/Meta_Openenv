"""Unit tests for NgSpiceSimulator and the mock simulator."""

from __future__ import annotations

import numpy as np
import pytest

from circuitsynth.simulator import NgSpiceSimulator, SimResult, _parse_ngspice_print


# ---------------------------------------------------------------------------
# SimResult API
# ---------------------------------------------------------------------------

class TestSimResult:
    def test_failure_factory(self):
        r = SimResult.failure("test error", sim_time=0.5)
        assert not r.success
        assert not r.convergence
        assert r.error_msg == "test error"
        assert r.sim_time_seconds == pytest.approx(0.5)
        assert r.time_array is None
        assert r.voltage_array is None

    def test_convergence_failure_factory(self):
        r = SimResult.convergence_failure()
        assert not r.success
        assert "convergence" in r.error_msg.lower()

    def test_to_dict(self):
        r = SimResult.failure("err")
        d = r.to_dict()
        assert "success" in d
        assert "convergence" in d
        assert "n_time_points" in d


# ---------------------------------------------------------------------------
# ngspice stdout parser
# ---------------------------------------------------------------------------

SAMPLE_OUTPUT = """\
ngspice 39 -> Circuit: Test
Index       time            v(n1)
  0    0.000000e+00    5.000000e+00
  1    1.000000e-06    4.999000e+00
  2    2.000000e-06    4.998000e+00
  3    3.000000e-06    4.997000e+00
  4    4.000000e-06    0.001000e+00
  5    5.000000e-06    0.002000e+00
  6    6.000000e-06    0.003000e+00
  7    7.000000e-06    4.996000e+00
  8    8.000000e-06    4.995000e+00
  9    9.000000e-06    4.994000e+00
 10    1.000000e-05    0.001000e+00
"""

class TestParseNgSpice:
    def test_parses_valid_output(self):
        result = _parse_ngspice_print(SAMPLE_OUTPUT)
        assert result is not None
        t, v = result
        assert len(t) == 11
        assert len(v) == 11
        assert t[0] == pytest.approx(0.0)
        assert v[0] == pytest.approx(5.0)

    def test_returns_none_for_empty(self):
        result = _parse_ngspice_print("")
        assert result is None

    def test_returns_none_for_no_data(self):
        result = _parse_ngspice_print("ngspice circuit error\n")
        assert result is None

    def test_handles_extra_header_lines(self):
        extra = "Some preamble\n" + SAMPLE_OUTPUT
        result = _parse_ngspice_print(extra)
        assert result is not None


# ---------------------------------------------------------------------------
# Mock simulator
# ---------------------------------------------------------------------------

class TestMockSimulator:
    @pytest.fixture
    def mock_sim(self):
        return NgSpiceSimulator(mock=True)

    def test_empty_circuit_returns_dc_or_noise(self, mock_sim):
        r = mock_sim.run_transient(
            netlist_str="* empty\n.END\n",
            stop_time=10e-3, step_size=5e-6,
            netlist_dict={"components": []},
            target_freq=1000.0,
        )
        assert r.success
        assert r.time_array is not None

    def test_good_circuit_returns_square_wave(self, mock_sim):
        mock_sim.set_rng(np.random.default_rng(0))
        net_dict = {
            "components": [
                {"comp_type": "NPN_BJT",   "value": 0.0},
                {"comp_type": "NPN_BJT",   "value": 0.0},
                {"comp_type": "RESISTOR",  "value": 47000.0},
                {"comp_type": "RESISTOR",  "value": 47000.0},
                {"comp_type": "CAPACITOR", "value": 15e-9},
                {"comp_type": "CAPACITOR", "value": 15e-9},
                {"comp_type": "VSOURCE",   "value": 5.0},
            ]
        }
        r = mock_sim.run_transient(
            netlist_str="* placeholder\n",
            stop_time=20e-3, step_size=5e-6,
            netlist_dict=net_dict,
            target_freq=1000.0,
        )
        assert r.success
        assert r.time_array is not None
        assert len(r.time_array) > 100
        # With a "good enough" circuit, the waveform should oscillate
        v = r.voltage_array
        vpp = v.max() - v.min()
        assert vpp > 0.5, f"Expected oscillation, got Vpp={vpp:.3f}"

    def test_result_time_step_correct(self, mock_sim):
        r = mock_sim.run_transient(
            netlist_str="*\n.END\n",
            stop_time=1e-3, step_size=1e-6,
            netlist_dict={"components": []},
        )
        if r.time_array is not None and len(r.time_array) > 1:
            dt = float(np.mean(np.diff(r.time_array)))
            assert abs(dt - 1e-6) / 1e-6 < 0.1, f"Time step mismatch: {dt}"

    def test_deterministic_with_same_rng(self, mock_sim):
        net = {"components": [{"comp_type": "NPN_BJT", "value": 0.0},
                               {"comp_type": "NPN_BJT", "value": 0.0},
                               {"comp_type": "CAPACITOR", "value": 10e-9},
                               {"comp_type": "RESISTOR", "value": 47e3},
                               {"comp_type": "VSOURCE", "value": 5.0}]}
        mock_sim.set_rng(np.random.default_rng(42))
        r1 = mock_sim.run_transient("*\n", 10e-3, 5e-6, net, 1000.0)
        mock_sim.set_rng(np.random.default_rng(42))
        r2 = mock_sim.run_transient("*\n", 10e-3, 5e-6, net, 1000.0)
        if r1.voltage_array is not None and r2.voltage_array is not None:
            np.testing.assert_array_equal(r1.voltage_array, r2.voltage_array)
