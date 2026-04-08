"""
Component library for the CircuitSynth RL environment.

Defines all allowable component types, their discrete value sets, pin names,
and SPICE model strings. These are the building blocks available to the agent.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Dict, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Component type enumeration
# ---------------------------------------------------------------------------

class ComponentType(IntEnum):
    RESISTOR  = 0
    CAPACITOR = 1
    NPN_BJT   = 2
    DIODE     = 3
    VSOURCE   = 4
    SWITCH    = 5
    NONE      = 6  # sentinel / padding


N_COMPONENT_TYPES = 6          # exclude NONE
N_VALUE_STEPS     = 20         # discrete levels per component type
N_NODES           = 12         # VCC, GND, N1..N10
MAX_COMPONENTS    = 12         # hard upper limit across all tasks

# ---------------------------------------------------------------------------
# Named node vocabulary
# ---------------------------------------------------------------------------

NODE_NAMES: List[str] = ["VCC", "GND"] + [f"N{i}" for i in range(1, 11)]
NODE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(NODE_NAMES)}
# SPICE maps GND → node 0 (special SPICE ground); VCC stays "VCC"
NODE_SPICE: Dict[str, str] = {
    "VCC": "VCC",
    "GND": "0",
    **{f"N{i}": f"N{i}" for i in range(1, 11)},
}

# ---------------------------------------------------------------------------
# Discrete value tables
# ---------------------------------------------------------------------------

RESISTOR_VALUES  = np.logspace(1, 7, N_VALUE_STEPS)     # 10 Ω  … 10 MΩ
CAPACITOR_VALUES = np.logspace(-12, -3, N_VALUE_STEPS)   # 1 pF  … 1 mF
VSOURCE_VALUES   = np.linspace(1.0, 15.0, N_VALUE_STEPS) # 1 V   … 15 V
DUMMY_VALUES     = np.zeros(N_VALUE_STEPS, dtype=float)   # fixed-model components

# ---------------------------------------------------------------------------
# Bundled SPICE models (public domain)
# ---------------------------------------------------------------------------

BJT_2N2222_MODEL = """\
.MODEL 2N2222 NPN(
+ IS=1.8108E-14 BF=200    NF=1     VAF=200   IKF=0.5   ISE=7.06E-14
+ NE=1.5        BR=3.29   NR=1     VAR=200   IKR=0.225 ISC=1.01E-12
+ NC=1.5        RB=10     IRB=0.1  RBM=1     RE=1      RC=1
+ XTB=1.5       EG=1.11   XTI=3
+ CJE=25E-12    VJE=0.7   MJE=0.3  TF=0.3E-9 XTF=2     VTF=6
+ ITF=0.1       CJC=11E-12 VJC=0.75 MJC=0.35 XCJC=0.9  TR=100E-9)
"""

DIODE_1N4148_MODEL = """\
.MODEL 1N4148 D(
+ IS=2.52E-9  RS=0.568  N=1.752  CJO=4E-12  M=0.4
+ TT=20E-9    BV=100    IBV=100E-6  VJ=0.4)
"""

SWITCH_MODEL = """\
.MODEL IDEAL_SW SW(RON=0.01 ROFF=1E9 VT=0.5 VH=0.1)
"""


# ---------------------------------------------------------------------------
# ComponentSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ComponentSpec:
    """Specification for one component type in the library."""
    comp_type:       ComponentType
    name:            str
    n_terminals:     int
    terminal_names:  List[str]
    value_range:     Tuple[float, float]
    values:          np.ndarray          # discrete allowed values
    spice_prefix:    str
    spice_model_name: Optional[str]
    spice_model_def:  Optional[str]
    description:     str
    unit:            str

    # ------------------------------------------------------------------
    def get_value(self, value_idx: int) -> float:
        """Return the actual physical value for a discrete index."""
        idx = int(np.clip(value_idx, 0, len(self.values) - 1))
        return float(self.values[idx])

    # ------------------------------------------------------------------
    def format_spice_value(self, value: float) -> str:
        """Format a value as a SPICE-compatible string."""
        if self.comp_type == ComponentType.RESISTOR:
            if value >= 1e6:
                return f"{value / 1e6:.4g}Meg"
            if value >= 1e3:
                return f"{value / 1e3:.4g}k"
            return f"{value:.4g}"
        if self.comp_type == ComponentType.CAPACITOR:
            if value >= 1e-3:
                return f"{value * 1e3:.4g}m"
            if value >= 1e-6:
                return f"{value * 1e6:.4g}u"
            if value >= 1e-9:
                return f"{value * 1e9:.4g}n"
            if value >= 1e-12:
                return f"{value * 1e12:.4g}p"
            return f"{value:.4e}"
        if self.comp_type == ComponentType.VSOURCE:
            return f"{value:.4g}"
        return str(value)

    # ------------------------------------------------------------------
    def feature_vector(self, value: float) -> np.ndarray:
        """
        Return a normalized feature vector for this component/value.
        Shape: (8,)  →  [type_one_hot(6), log_value_norm, is_model_based]
        """
        oh = np.zeros(N_COMPONENT_TYPES, dtype=np.float32)
        oh[int(self.comp_type)] = 1.0

        vmin, vmax = self.value_range
        if vmax > vmin and vmin > 0:
            log_v = np.log10(max(value, 1e-20))
            log_min = np.log10(vmin)
            log_max = np.log10(vmax)
            norm_val = float(np.clip((log_v - log_min) / (log_max - log_min), 0.0, 1.0))
        else:
            norm_val = 0.0

        is_model = float(self.spice_model_name is not None)
        return np.array([*oh, norm_val, is_model], dtype=np.float32)


# ---------------------------------------------------------------------------
# Component library dictionary
# ---------------------------------------------------------------------------

COMPONENT_LIBRARY: Dict[ComponentType, ComponentSpec] = {
    ComponentType.RESISTOR: ComponentSpec(
        comp_type=ComponentType.RESISTOR,
        name="Resistor",
        n_terminals=2,
        terminal_names=["A", "B"],
        value_range=(10.0, 10e6),
        values=RESISTOR_VALUES,
        spice_prefix="R",
        spice_model_name=None,
        spice_model_def=None,
        description="Ideal resistor",
        unit="Ω",
    ),
    ComponentType.CAPACITOR: ComponentSpec(
        comp_type=ComponentType.CAPACITOR,
        name="Capacitor",
        n_terminals=2,
        terminal_names=["A", "B"],
        value_range=(1e-12, 1e-3),
        values=CAPACITOR_VALUES,
        spice_prefix="C",
        spice_model_name=None,
        spice_model_def=None,
        description="Ideal capacitor",
        unit="F",
    ),
    ComponentType.NPN_BJT: ComponentSpec(
        comp_type=ComponentType.NPN_BJT,
        name="NPN BJT (2N2222)",
        n_terminals=3,
        terminal_names=["C", "B", "E"],
        value_range=(0.0, 0.0),
        values=DUMMY_VALUES,
        spice_prefix="Q",
        spice_model_name="2N2222",
        spice_model_def=BJT_2N2222_MODEL,
        description="2N2222 NPN bipolar junction transistor",
        unit="—",
    ),
    ComponentType.DIODE: ComponentSpec(
        comp_type=ComponentType.DIODE,
        name="Diode (1N4148)",
        n_terminals=2,
        terminal_names=["A", "K"],
        value_range=(0.0, 0.0),
        values=DUMMY_VALUES,
        spice_prefix="D",
        spice_model_name="1N4148",
        spice_model_def=DIODE_1N4148_MODEL,
        description="1N4148 fast switching diode",
        unit="—",
    ),
    ComponentType.VSOURCE: ComponentSpec(
        comp_type=ComponentType.VSOURCE,
        name="DC Voltage Source",
        n_terminals=2,
        terminal_names=["+", "-"],
        value_range=(1.0, 15.0),
        values=VSOURCE_VALUES,
        spice_prefix="V",
        spice_model_name=None,
        spice_model_def=None,
        description="Ideal DC voltage source",
        unit="V",
    ),
    ComponentType.SWITCH: ComponentSpec(
        comp_type=ComponentType.SWITCH,
        name="Ideal Switch",
        n_terminals=2,
        terminal_names=["A", "B"],
        value_range=(0.0, 1.0),
        values=DUMMY_VALUES,
        spice_prefix="S",
        spice_model_name="IDEAL_SW",
        spice_model_def=SWITCH_MODEL,
        description="Ideal voltage-controlled switch",
        unit="—",
    ),
}

# Ordered list for indexing
COMPONENT_TYPE_LIST: List[ComponentType] = [
    ComponentType.RESISTOR,
    ComponentType.CAPACITOR,
    ComponentType.NPN_BJT,
    ComponentType.DIODE,
    ComponentType.VSOURCE,
    ComponentType.SWITCH,
]
