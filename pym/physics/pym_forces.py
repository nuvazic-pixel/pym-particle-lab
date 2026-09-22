from __future__ import annotations
import numpy as np
from pym.core.state import PYMState


def memory_force(state: PYMState, coupling_scale: float = 0.1) -> np.ndarray:
    """Minimal v0.1 hypothesis: memory biases the first spatial axis."""
    force = np.zeros_like(state.position, dtype=float)
    if force.size:
        force[0] = coupling_scale * state.memory
    return force
