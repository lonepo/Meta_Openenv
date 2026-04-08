"""
Netlist graph manager for CircuitSynth.

Maintains a list of placed components with their node connections. Provides:
  - add_component / remove_component
  - Circuit validation (connectivity, short-circuit, ground, power)
  - Serialization to SPICE .cir format
  - Adjacency matrix and component feature matrix for RL observations
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .components import (
    COMPONENT_LIBRARY,
    MAX_COMPONENTS,
    N_NODES,
    NODE_NAMES,
    NODE_SPICE,
    ComponentType,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlacedComponent:
    """A single component placed in the circuit."""
    comp_id: str                    # e.g. "R1", "C3", "Q1"
    comp_type: ComponentType
    value: float                    # physical value (Ω, F, V, …)
    value_idx: int                  # discrete index into spec.values
    connections: Dict[str, int]     # terminal_name → node_index (into NODE_NAMES)

    def to_dict(self) -> dict:
        return {
            "comp_id": self.comp_id,
            "comp_type": self.comp_type.name,
            "value": self.value,
            "value_idx": self.value_idx,
            "connections": {t: NODE_NAMES[n] for t, n in self.connections.items()},
        }


@dataclass
class ActionResult:
    """Result of applying an action to the netlist."""
    success: bool
    comp_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Outcome of netlist validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_ground: bool = False
    has_power: bool = False
    is_connected: bool = False
    has_short_circuit: bool = False
    n_components: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Netlist class
# ---------------------------------------------------------------------------

class Netlist:
    """
    Mutable circuit graph used by the RL agent.

    Nodes are identified by index (0 = VCC, 1 = GND, 2..11 = N1..N10).
    SPICE ground is always node 0 in ngspice; NODE_SPICE maps our GND → "0".
    """

    def __init__(self) -> None:
        self.components: List[PlacedComponent] = []
        self._type_counters: Dict[ComponentType, int] = {ct: 0 for ct in ComponentType}
        self._id_counter = itertools.count(1)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._id_counter)}"

    def add_component(
        self,
        comp_type: ComponentType,
        value: float,
        value_idx: int,
        connections: Dict[str, int],
    ) -> ActionResult:
        """
        Add a component to the circuit.

        Parameters
        ----------
        comp_type : ComponentType
        value     : physical value
        value_idx : discrete value index
        connections : {terminal_name: node_index}

        Returns
        -------
        ActionResult with success flag and assigned comp_id.
        """
        if len(self.components) >= MAX_COMPONENTS:
            return ActionResult(success=False, error="MAX_COMPONENTS budget exceeded")

        spec = COMPONENT_LIBRARY.get(comp_type)
        if spec is None or comp_type == ComponentType.NONE:
            return ActionResult(success=False, error=f"Unknown component type: {comp_type}")

        # Validate terminal names
        for term in connections:
            if term not in spec.terminal_names:
                return ActionResult(success=False,
                                    error=f"Invalid terminal '{term}' for {spec.name}")

        # Validate node indices
        for term, node_idx in connections.items():
            if not (0 <= node_idx < N_NODES):
                return ActionResult(success=False,
                                    error=f"Node index {node_idx} out of range [0, {N_NODES})")

        # Reject degenerate 2-terminal placements (A==B → short)
        if spec.n_terminals == 2:
            nodes = list(connections.values())
            if nodes[0] == nodes[1]:
                return ActionResult(success=False,
                                    error="Two-terminal component cannot connect same node to both pins")

        # Reject degenerate BJT placements
        if comp_type == ComponentType.NPN_BJT:
            nodes = list(connections.values())
            if len(set(nodes)) < 3:
                return ActionResult(success=False,
                                    error="BJT terminals C, B, E must connect to distinct nodes")

        prefx = spec.spice_prefix
        comp_id = self._next_id(prefx)
        self._type_counters[comp_type] = self._type_counters.get(comp_type, 0) + 1

        pc = PlacedComponent(
            comp_id=comp_id,
            comp_type=comp_type,
            value=value,
            value_idx=value_idx,
            connections=connections,
        )
        self.components.append(pc)
        return ActionResult(success=True, comp_id=comp_id)

    def remove_component_by_index(self, idx: int) -> ActionResult:
        """Remove the component at list index idx."""
        if idx < 0 or idx >= len(self.components):
            return ActionResult(success=False,
                                error=f"No component at index {idx} (len={len(self.components)})")
        removed = self.components.pop(idx)
        self._type_counters[removed.comp_type] = max(
            0, self._type_counters.get(removed.comp_type, 0) - 1
        )
        return ActionResult(success=True, comp_id=removed.comp_id)

    def clear(self) -> None:
        """Remove all components."""
        self.components.clear()
        self._type_counters = {ct: 0 for ct in ComponentType}

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(N_NODES))
        for pc in self.components:
            nodes = list(pc.connections.values())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    G.add_edge(nodes[i], nodes[j])
        return G

    def used_nodes(self) -> set:
        used = set()
        for pc in self.components:
            used.update(pc.connections.values())
        return used

    def count(self, comp_type: ComponentType) -> int:
        return sum(1 for c in self.components if c.comp_type == comp_type)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """
        Check the circuit for basic physical validity before SPICE submission.

        Rules:
          1. Must have at least one component.
          2. Must have GND (node index 1) used somewhere.
          3. Must have a voltage source connected, or at least VCC (node 0).
          4. No component may directly short VCC (node 0) to GND (node 1)
             without impedance (a resistor/cap between them is fine, a wire is not).
          5. Circuit must be a connected graph across used nodes.
        """
        errors: List[str] = []
        warnings: List[str] = []

        n_comps = len(self.components)
        if n_comps == 0:
            return ValidationResult(valid=False, errors=["No components placed"])

        used = self.used_nodes()

        # --- ground check ---
        has_ground = (1 in used)  # node 1 = GND
        if not has_ground:
            errors.append("No component connected to GND node")

        # --- power check ---
        has_vsource = self.count(ComponentType.VSOURCE) > 0
        has_vcc = (0 in used)  # node 0 = VCC
        has_power = has_vsource or has_vcc
        if not has_power:
            errors.append("No voltage source or VCC connection in circuit")

        # --- short-circuit check (VCC directly tied to GND) ---
        has_short = False
        for pc in self.components:
            nodes = list(pc.connections.values())
            if {0, 1}.issubset(set(nodes)) and pc.comp_type == ComponentType.VSOURCE:
                # A VSOURCE between VCC and GND is fine — that's the supply
                continue
            if pc.comp_type not in (ComponentType.RESISTOR, ComponentType.CAPACITOR,
                                     ComponentType.DIODE, ComponentType.SWITCH):
                # Only passive 2-terminal components should be between VCC and GND directly
                pass
            # Flag if any component (other than source) directly shorts vcc to gnd
            if 0 in nodes and 1 in nodes and pc.comp_type == ComponentType.SWITCH:
                errors.append(f"Switch {pc.comp_id} directly shorts VCC to GND")
                has_short = True

        # --- connectivity check (only over used nodes) ---
        if len(used) > 1:
            G = self._build_graph()
            subgraph = G.subgraph(used)
            is_connected = nx.is_connected(subgraph)
        else:
            is_connected = (n_comps == 1)

        if not is_connected:
            errors.append("Circuit graph is not fully connected — floating nodes detected")

        # --- mild warnings ---
        if self.count(ComponentType.NPN_BJT) == 0:
            warnings.append("No BJT placed — circuit may not oscillate")
        if self.count(ComponentType.CAPACITOR) == 0:
            warnings.append("No capacitor placed — timing elements missing")

        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            has_ground=has_ground,
            has_power=has_power,
            is_connected=is_connected,
            has_short_circuit=has_short,
            n_components=n_comps,
        )

    # ------------------------------------------------------------------
    # SPICE serialization
    # ------------------------------------------------------------------

    def to_spice(self, output_node: str, stop_time: float, step_size: float) -> str:
        """
        Serialize the current circuit to a SPICE netlist string.

        Parameters
        ----------
        output_node : str  — node name to probe (e.g. "N1")
        stop_time   : float — transient simulation stop time (seconds)
        step_size   : float — time step (seconds)
        """
        lines: List[str] = []
        lines.append("* CircuitSynth Auto-Generated Netlist")
        lines.append("* ----------------------------------------")

        # Simulation options for convergence robustness
        lines.append(".OPTIONS ABSTOL=1e-9 RELTOL=1e-3 VNTOL=1e-6 "
                     "ITL1=1000 ITL2=500 ITL4=100 METHOD=GEAR")
        lines.append("")

        # Collect needed model definitions
        models_needed: Dict[str, str] = {}
        for pc in self.components:
            spec = COMPONENT_LIBRARY[pc.comp_type]
            if spec.spice_model_name and spec.spice_model_def:
                models_needed[spec.spice_model_name] = spec.spice_model_def

        for model_def in models_needed.values():
            lines.append(model_def.strip())
            lines.append("")

        # Component lines
        for pc in self.components:
            spec = COMPONENT_LIBRARY[pc.comp_type]
            node_str = " ".join(
                NODE_SPICE.get(NODE_NAMES[n], NODE_NAMES[n])
                for n in pc.connections.values()
            )

            if pc.comp_type == ComponentType.NPN_BJT:
                line = f"{pc.comp_id} {node_str} {spec.spice_model_name}"
            elif pc.comp_type == ComponentType.DIODE:
                line = f"{pc.comp_id} {node_str} {spec.spice_model_name}"
            elif pc.comp_type == ComponentType.VSOURCE:
                val_str = spec.format_spice_value(pc.value)
                line = f"{pc.comp_id} {node_str} DC {val_str}"
            elif pc.comp_type == ComponentType.SWITCH:
                # Ideal switch needs a control voltage — tie to VCC for "always on"
                line = f"{pc.comp_id} {node_str} VCC 0 {spec.spice_model_name}"
            else:
                val_str = spec.format_spice_value(pc.value)
                line = f"{pc.comp_id} {node_str} {val_str}"

            lines.append(line)

        lines.append("")

        # Initial conditions to help convergence: perturb one node slightly
        ic_nodes = [pc.connections for pc in self.components
                    if pc.comp_type == ComponentType.CAPACITOR]
        if ic_nodes:
            ic_parts = []
            for connections in ic_nodes[:2]:
                node_idx = list(connections.values())[0]
                spice_node = NODE_SPICE.get(NODE_NAMES[node_idx], NODE_NAMES[node_idx])
                ic_parts.append(f"V({spice_node})=0.1")
            lines.append(f".IC {' '.join(ic_parts)}")

        # Transient simulation command
        lines.append(f".TRAN {step_size:.4e} {stop_time:.4e} UIC")

        # Output probe
        out_spice = NODE_SPICE.get(output_node, output_node)
        lines.append(f".PRINT TRAN V({out_spice})")

        lines.append(".END")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def to_adjacency_matrix(self, n_nodes: int = N_NODES) -> np.ndarray:
        """
        Return a (n_nodes, n_nodes) float32 adjacency matrix.
        Entry [i, j] = number of components connecting node i and node j.
        """
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for pc in self.components:
            nodes = list(pc.connections.values())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    ni, nj = nodes[i], nodes[j]
                    if 0 <= ni < n_nodes and 0 <= nj < n_nodes:
                        adj[ni, nj] += 1.0
                        adj[nj, ni] += 1.0
        # Normalize by max to [0, 1]
        mx = adj.max()
        if mx > 0:
            adj /= mx
        return adj

    def get_component_features(self, max_components: int = MAX_COMPONENTS) -> np.ndarray:
        """
        Return a (max_components, 8) float32 feature matrix.
        Each row: [type_one_hot(6), norm_value, is_model_component]
        Vacant slots are zero-padded.
        """
        feat_dim = 8  # 6 one-hot + 1 norm_value + 1 is_model
        mat = np.zeros((max_components, feat_dim), dtype=np.float32)
        for i, pc in enumerate(self.components[:max_components]):
            spec = COMPONENT_LIBRARY[pc.comp_type]
            mat[i] = spec.feature_vector(pc.value)
        return mat

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_components": len(self.components),
            "components": [c.to_dict() for c in self.components],
            "used_nodes": sorted(self.used_nodes()),
        }

    def __len__(self) -> int:
        return len(self.components)

    def __repr__(self) -> str:
        parts = [f"{c.comp_id}({c.comp_type.name})" for c in self.components]
        return f"Netlist([{', '.join(parts)}])"
