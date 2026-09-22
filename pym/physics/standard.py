from __future__ import annotations
import numpy as np
from pym.core.state import PYMState


def harmonic_force(state: PYMState, k: float = 0.5) -> np.ndarray:
    return -k * state.position


def harmonic_energy(state: PYMState, k: float = 0.5) -> float:
    kinetic = float(np.dot(state.momentum, state.momentum) / (2.0 * state.mass))
    potential = float(0.5 * k * np.dot(state.position, state.position))
    return kinetic + potential
