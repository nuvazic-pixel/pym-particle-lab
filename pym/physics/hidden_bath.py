from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class HiddenBathState:
    q: np.ndarray
    p: np.ndarray
    y: np.ndarray
    py: np.ndarray
    mass: float = 1.0
    bath_mass: float = 1.0

    def clone(self) -> "HiddenBathState":
        return HiddenBathState(self.q.copy(), self.p.copy(), self.y.copy(), self.py.copy(), self.mass, self.bath_mass)


def total_energy(state: HiddenBathState, k_ext: float = 1.0, k_bath: float = 0.5) -> float:
    particle_ke = np.dot(state.p, state.p) / (2.0 * state.mass)
    bath_ke = np.dot(state.py, state.py) / (2.0 * state.bath_mass)
    ext_pe = 0.5 * k_ext * np.dot(state.q, state.q)
    coupling_pe = 0.5 * k_bath * np.dot(state.q - state.y, state.q - state.y)
    return float(particle_ke + bath_ke + ext_pe + coupling_pe)


def step_velocity_verlet(state: HiddenBathState, dt: float, k_ext: float = 1.0, k_bath: float = 0.5) -> HiddenBathState:
    """Conservative two-oscillator comparator. No damping in the conservation-gate model."""
    s = state.clone()
    fq = -k_ext * s.q - k_bath * (s.q - s.y)
    fy = -k_bath * (s.y - s.q)
    p_half = s.p + 0.5 * dt * fq
    py_half = s.py + 0.5 * dt * fy
    q_new = s.q + dt * p_half / s.mass
    y_new = s.y + dt * py_half / s.bath_mass
    fq_new = -k_ext * q_new - k_bath * (q_new - y_new)
    fy_new = -k_bath * (y_new - q_new)
    s.q = q_new
    s.y = y_new
    s.p = p_half + 0.5 * dt * fq_new
    s.py = py_half + 0.5 * dt * fy_new
    return s
