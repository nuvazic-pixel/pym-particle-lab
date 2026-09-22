from __future__ import annotations
import numpy as np
from .state import PYMState


def interaction_signal(state: PYMState, force_standard: np.ndarray, dt: float) -> float:
    """Neutral v0.1 signal. It is NOT asserted to be physical information."""
    return float(np.linalg.norm(force_standard) * dt)


def update_memory(memory: float, signal: float, alpha: float) -> float:
    return float(alpha * memory + (1.0 - alpha) * signal)
