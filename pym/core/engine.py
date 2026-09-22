from __future__ import annotations
from collections.abc import Callable
import numpy as np
from .memory import interaction_signal, update_memory
from .state import PYMState

ForceFn = Callable[[PYMState], np.ndarray]


class PYMEngine:
    """Deterministic Euler-Cromer v0.1 integrator."""

    def step(self, state: PYMState, dt: float, standard_force: ForceFn, pym_force: ForceFn) -> PYMState:
        if dt <= 0:
            raise ValueError("dt must be positive")

        f_std = np.asarray(standard_force(state), dtype=float)
        f_pym = np.asarray(pym_force(state), dtype=float)
        if f_std.shape != state.position.shape or f_pym.shape != state.position.shape:
            raise ValueError("force shape must match state shape")

        total_force = f_std + state.lambda_pym * f_pym
        new_momentum = state.momentum + total_force * dt
        new_position = state.position + (new_momentum / state.mass) * dt

        signal = interaction_signal(state, f_std, dt)
        new_memory = update_memory(state.memory, signal, state.alpha)

        out = state.clone()
        out.position = new_position
        out.momentum = new_momentum
        out.interaction_signal = signal
        out.memory = new_memory
        return out
