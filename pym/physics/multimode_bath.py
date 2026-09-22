from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BathParameters:
    mass: np.ndarray
    omega: np.ndarray
    coupling: np.ndarray

    @property
    def n_modes(self) -> int:
        return len(self.mass)


@dataclass
class MultiModeState:
    q: np.ndarray
    p: np.ndarray
    y: np.ndarray
    py: np.ndarray
    particle_mass: float = 1.0

    def clone(self) -> "MultiModeState":
        return MultiModeState(self.q.copy(), self.p.copy(), self.y.copy(), self.py.copy(), self.particle_mass)


def forces(state: MultiModeState, params: BathParameters, k_ext: float = 1.0):
    # Caldeira-Leggett-like shifted oscillators:
    # V_i = .5*m_i*omega_i^2*(y_i - c_i*q/(m_i*omega_i^2))^2
    denom = params.mass * params.omega**2
    shift = params.coupling[:, None] * state.q[None, :] / denom[:, None]
    bath_delta = state.y - shift
    f_y = -(params.mass * params.omega**2)[:, None] * bath_delta
    f_q = -k_ext * state.q + np.sum(params.coupling[:, None] * bath_delta, axis=0)
    return f_q, f_y


def total_energy(state: MultiModeState, params: BathParameters, k_ext: float = 1.0) -> float:
    particle_ke = np.dot(state.p, state.p) / (2.0 * state.particle_mass)
    ext_pe = 0.5 * k_ext * np.dot(state.q, state.q)
    bath_ke = np.sum(np.sum(state.py**2, axis=1) / (2.0 * params.mass))
    denom = params.mass * params.omega**2
    shift = params.coupling[:, None] * state.q[None, :] / denom[:, None]
    bath_pe = 0.5 * np.sum((params.mass * params.omega**2)[:, None] * (state.y - shift)**2)
    return float(particle_ke + ext_pe + bath_ke + bath_pe)


def step_velocity_verlet(state: MultiModeState, params: BathParameters, dt: float, k_ext: float = 1.0) -> MultiModeState:
    s = state.clone()
    fq, fy = forces(s, params, k_ext)
    p_half = s.p + 0.5 * dt * fq
    py_half = s.py + 0.5 * dt * fy
    q_new = s.q + dt * p_half / s.particle_mass
    y_new = s.y + dt * py_half / params.mass[:, None]
    trial = MultiModeState(q_new, p_half, y_new, py_half, s.particle_mass)
    fq2, fy2 = forces(trial, params, k_ext)
    s.q, s.y = q_new, y_new
    s.p = p_half + 0.5 * dt * fq2
    s.py = py_half + 0.5 * dt * fy2
    return s


def simulate(params: BathParameters, q0: np.ndarray, p0: np.ndarray, steps: int, dt: float, k_ext: float = 1.0):
    n = params.n_modes
    state = MultiModeState(
        np.asarray(q0, float).copy(),
        np.asarray(p0, float).copy(),
        np.zeros((n, len(q0))),
        np.zeros((n, len(q0))),
    )
    qs, ps = [], []
    e0 = total_energy(state, params, k_ext)
    for _ in range(steps):
        state = step_velocity_verlet(state, params, dt, k_ext)
        qs.append(state.q.copy())
        ps.append(state.p.copy())
    drift = abs(total_energy(state, params, k_ext) - e0) / (abs(e0) + 1e-15)
    return np.asarray(qs), np.asarray(ps), drift
