"""
Structured typed action space for CircuitSynth.

Every action the agent can take is encoded as a fixed-length integer vector
compatible with gymnasium.spaces.MultiDiscrete.

Action vector layout (7 integers):
  [0] action_type   : 0=ADD_COMPONENT, 1=REMOVE_COMPONENT, 2=FINALIZE, 3=NO_OP
  [1] component_type: 0=RESISTOR, 1=CAPACITOR, 2=NPN_BJT, 3=DIODE,
                       4=VSOURCE, 5=SWITCH, 6=NONE
  [2] value_idx     : 0–19 (discrete index into component value table)
  [3] node_a        : 0–11 (VCC, GND, N1..N10)
  [4] node_b        : 0–11
  [5] node_c        : 0–11 (third terminal for BJT; 0=VCC=unused sentinel otherwise)
  [6] remove_idx    : 0–11 (which placed component to remove; only for REMOVE)

MultiDiscrete shape: [4, 7, 20, 12, 12, 12, 12]
Total combinations: 4 × 7 × 20 × 12 × 12 × 12 × 12 = ~11.6 M  (finite, tractable)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import numpy as np

from .components import (
    COMPONENT_LIBRARY,
    N_NODES,
    N_VALUE_STEPS,
    ComponentType,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    ADD_COMPONENT    = 0
    REMOVE_COMPONENT = 1
    FINALIZE         = 2
    NO_OP            = 3


# MultiDiscrete n-values for each dimension
ACTION_NVEC = np.array([
    len(ActionType),     # 4  — action type
    7,                   # 7  — component type (incl. NONE)
    N_VALUE_STEPS,       # 20 — value index
    N_NODES,             # 12 — node A
    N_NODES,             # 12 — node B
    N_NODES,             # 12 — node C (BJT emitter)
    12,                  # 12 — remove_idx
], dtype=np.int64)

ACTION_DIM = len(ACTION_NVEC)


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """Typed action for one agent decision."""
    action_type:    ActionType
    component_type: ComponentType = ComponentType.NONE
    value_idx:      int           = 0
    node_a:         int           = 0    # GND for FINALIZE / NO_OP
    node_b:         int           = 1    # GND for FINALIZE / NO_OP
    node_c:         int           = 1    # GND for 2-terminal components
    remove_idx:     int           = 0

    # ------------------------------------------------------------------
    @classmethod
    def decode(cls, vec: np.ndarray) -> "Action":
        """Decode a raw integer array from the RL policy into a typed Action."""
        vec = np.asarray(vec, dtype=np.int64).ravel()
        if len(vec) < ACTION_DIM:
            vec = np.pad(vec, (0, ACTION_DIM - len(vec)), constant_values=0)

        action_type    = ActionType(int(np.clip(vec[0], 0, len(ActionType) - 1)))
        component_type = ComponentType(int(np.clip(vec[1], 0, 6)))
        value_idx      = int(np.clip(vec[2], 0, N_VALUE_STEPS - 1))
        node_a         = int(np.clip(vec[3], 0, N_NODES - 1))
        node_b         = int(np.clip(vec[4], 0, N_NODES - 1))
        node_c         = int(np.clip(vec[5], 0, N_NODES - 1))
        remove_idx     = int(np.clip(vec[6], 0, 11))

        return cls(
            action_type=action_type,
            component_type=component_type,
            value_idx=value_idx,
            node_a=node_a,
            node_b=node_b,
            node_c=node_c,
            remove_idx=remove_idx,
        )

    def encode(self) -> np.ndarray:
        """Encode this action as an integer array for storage / logging."""
        return np.array([
            int(self.action_type),
            int(self.component_type),
            self.value_idx,
            self.node_a,
            self.node_b,
            self.node_c,
            self.remove_idx,
        ], dtype=np.int64)

    # ------------------------------------------------------------------
    @classmethod
    def add(
        cls,
        comp_type: ComponentType,
        value_idx: int,
        node_a: int,
        node_b: int,
        node_c: int = 1,   # GND by default (harmless for 2-terminal)
    ) -> "Action":
        """Convenience constructor for ADD_COMPONENT actions."""
        return cls(ActionType.ADD_COMPONENT, comp_type, value_idx, node_a, node_b, node_c)

    @classmethod
    def remove(cls, remove_idx: int) -> "Action":
        return cls(ActionType.REMOVE_COMPONENT, remove_idx=remove_idx)

    @classmethod
    def finalize() -> "Action":
        return Action(ActionType.FINALIZE)

    @classmethod
    def noop(cls) -> "Action":
        return cls(ActionType.NO_OP)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        from .components import NODE_NAMES
        return {
            "action_type":    self.action_type.name,
            "component_type": self.component_type.name,
            "value_idx":      self.value_idx,
            "node_a":         NODE_NAMES[self.node_a],
            "node_b":         NODE_NAMES[self.node_b],
            "node_c":         NODE_NAMES[self.node_c],
            "remove_idx":     self.remove_idx,
        }

    def __repr__(self) -> str:
        return (
            f"Action({self.action_type.name}, "
            f"comp={self.component_type.name}, "
            f"val_idx={self.value_idx}, "
            f"nodes=[{self.node_a},{self.node_b},{self.node_c}], "
            f"rm_idx={self.remove_idx})"
        )


# ---------------------------------------------------------------------------
# Action masking helpers
# ---------------------------------------------------------------------------

@dataclass
class ActionMask:
    """
    Structured boolean mask over valid actions.

    For MultiDiscrete spaces, gymnasium's action masking protocol expects an
    array of shape (sum(nvec),) with 0/1 flags for each discrete value.
    """
    can_add:      bool = True
    can_remove:   bool = False
    can_finalize: bool = True
    can_noop:     bool = True

    # Which component types are allowed (e.g. None = max budget reached)
    allowed_comp_types: List[ComponentType] = None   # None = all

    def __post_init__(self):
        if self.allowed_comp_types is None:
            self.allowed_comp_types = [ct for ct in ComponentType if ct != ComponentType.NONE]

    def to_flat_mask(self) -> np.ndarray:
        """
        Return a flat boolean mask of shape (sum(ACTION_NVEC),) = (79,).
        Used by frameworks that support invalid-action masking (e.g. MaskablePPO).
        """
        mask = np.ones(int(ACTION_NVEC.sum()), dtype=bool)
        offset = 0

        # action_type mask (4)
        atype_mask = np.array([
            self.can_add, self.can_remove, self.can_finalize, self.can_noop
        ], dtype=bool)
        mask[offset: offset + 4] = atype_mask
        offset += 4

        # component_type mask (7)
        ctype_mask = np.zeros(7, dtype=bool)
        for ct in self.allowed_comp_types:
            ctype_mask[int(ct)] = True
        mask[offset: offset + 7] = ctype_mask
        offset += 7

        # value_idx (20), node_a (12), node_b (12), node_c (12), remove_idx (12)
        # All values allowed for these dimensions
        offset += 20 + 12 + 12 + 12 + 12

        return mask


def build_action_mask(
    n_placed: int,
    max_components: int,
) -> ActionMask:
    """Create an ActionMask given current circuit state."""
    budget_full = n_placed >= max_components
    return ActionMask(
        can_add=not budget_full,
        can_remove=n_placed > 0,
        can_finalize=True,
        can_noop=True,
        allowed_comp_types=(
            [] if budget_full
            else [ct for ct in ComponentType if ct != ComponentType.NONE]
        ),
    )
